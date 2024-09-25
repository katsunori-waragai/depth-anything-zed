"""
Depth Anything を用いて欠損点のdepthを補完する。

- Depth の単位は [mm] です。
- Depth Anything で算出したdisparity: 単位は不明。depth ∝ 1 / disparity
    近距離で値が大きい。

TODO：
- RANSACでの合わせ込みの際に、遠方側の点は除外しよう。
    - 遠方側で関係式を合わせようとすると誤差が大きくなるはずのため。
    - 視差とdepthとの関係式は depth ∝ 1/視差　であるべき。
    - log-log プロットで傾き　-1であるべき。
    - 遠方側は除外して、近距離用のRANSACでフィッティングを作ろう。
- logスケールでのfittingの残差を表示すること。
    - 残差の分だけ距離を間違えることになる。
- 補完後のdepthから、3DのpointCloud を得られるようにすること。
参考：zed sdk でのpoint_cloud の算出
```
point_cloud = sl.Mat()
zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
xyz_data = point_cloud.get_data()
```
-　
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sklearn
from matplotlib import pylab as plt

from depanyzed.depth2pointcloud import disparity_to_depth


def isfinite_near_pixels(
    zed_depth: np.ndarray, da_disparity: np.ndarray, far_depth_limit=5000, small_disparity_limit=math.exp(0.5)
):
    """
    RANSAC で合わせこみをする際に、事前に選択する画素をboolの配列として選択する。

    far_depth_limit [mm]
    small_disparity_limit [pixel]
    """
    isfinite_pixels = np.isfinite(zed_depth)
    isnear = np.less(zed_depth, far_depth_limit)  # [mm]
    isnear_da = np.greater(da_disparity, small_disparity_limit)
    isfinite_near = np.logical_and(isfinite_pixels, isnear)
    isfinite_near = np.logical_and(isfinite_near, isnear_da)
    return isfinite_near


@dataclass
class DepthComplementor:
    """
    保持するもの
    - fitした値の係数などの情報

    改変方針
    da_disparity:
        depth anything の算出する視差。しかし、単眼で算出しているので、絶対値は信頼ができない。
    inv_zed_depth:
        ZED SDK のdepthの逆数をとったもの。視差に比例する。

    predictの動作:
        inv_zed_depth = self.predict(da_disparity)
        の入出力とする。
    """

    use_fixed_model: bool = True
    EPS: float = 1e-6
    predictable: bool = False  # 最初のフィッティングがされないうちは、predict()できない。

    def __post_init__(self):
        if self.use_fixed_model:
            from depanyzed.fixed_intercept import FixedInterceptRegressor

            self.ransac = sklearn.linear_model.RANSACRegressor(
                estimator=FixedInterceptRegressor(), min_samples=2, residual_threshold=None, max_trials=1000
            )
        else:
            self.ransac = sklearn.linear_model.RANSACRegressor()

    def fit(self, da_disparity: np.ndarray, zed_disparity: np.ndarray, isfinite_near: np.ndarray, plot=True):
        """
        isfinite_near がtrue である画素について、zed sdkのdepthと depth anything の視差(disparity) との関係式を算出する。
        - RANSAC のアルゴリズムを使用。
        - 深度は視差に反比例します。
        - そのため、log(深度) とlog(視差) は-1の勾配で1次式で表されると期待します。
        """
        t0 = cv2.getTickCount()
        assert zed_disparity.shape[:2] == da_disparity.shape[:2]
        effective_zed_disparity = zed_disparity[isfinite_near]
        effective_da_disparity = da_disparity[isfinite_near]

        print(f"{np.max(effective_zed_disparity)=}")
        print(f"{np.max(effective_da_disparity)=}")
        X = np.asarray(effective_da_disparity)  # disparity
        Y = np.asarray(effective_zed_disparity)  # depth
        X = X.flatten().reshape(-1, 1)
        self.ransac.fit(X, Y)
        self.predictable = True
        inlier_mask = self.ransac.inlier_mask_
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in fit")
        predicted_Y = self.ransac.predict(X)
        if plot:
            self.regression_plot(X, Y, predicted_Y, inlier_mask, pngname=Path("data/depth_cmp_log.png"))

    def predict(self, da_disparity: np.ndarray) -> np.ndarray:
        """
        depth anything 由来の値をZED SDK でのdepthの逆数相当に変換する。
        """
        t0 = cv2.getTickCount()
        assert self.predictable
        r = self.ransac.predict(da_disparity)
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in predict")
        return r

    def regression_plot(
        self, X: np.ndarray, Y: np.ndarray, predicted_Y: np.ndarray, inlier_mask, pngname=Path("depth_cmp_log.png")
    ):
        plt.figure(1, figsize=(8, 6))
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(X, Y, ".")
        plt.plot(X, predicted_Y, ".")
        plt.xlabel("Depth-Anything disparity")
        plt.ylabel("ZED SDK disparity")
        plt.xlim(0, None)
        plt.ylim(0, None)
        plt.grid(True)
        plt.subplot(2, 2, 2)

        plt.plot(X[inlier_mask], Y[inlier_mask], ".")
        plt.plot(X, predicted_Y, ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity")
        plt.ylabel("ZED SDK disparity")
        plt.xlim(0, None)
        plt.ylim(0, None)
        plt.grid(True)
        plt.subplot(2, 2, 4)

        plt.plot(X[inlier_mask], Y[inlier_mask] - predicted_Y[inlier_mask], ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity")
        plt.ylabel("disparity difference ")
        plt.grid(True)
        plt.xlim(0, None)
        pngname.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(pngname)

    def complement(self, zed_depth: np.ndarray, da_disparity: np.ndarray):
        """
        input, output in linear scale
        return depth anything based depth
        """
        h, w = zed_depth.shape[:2]
        X_full = da_disparity.flatten().reshape(-1, 1)
        predicted_disparity = self.predict(X_full)
        predicted_disparity = np.maximum(predicted_disparity, 0.0)
        predicted_disparity = np.reshape(predicted_disparity, (h, w))
        predicted_depth = disparity_to_depth(predicted_disparity)
        mixed_depth = np.reshape(predicted_depth.copy(), (h, w))
        isfinite_near = isfinite_near_pixels(zed_depth, da_disparity)
        mixed_depth[isfinite_near] = zed_depth[isfinite_near]

        assert np.alltrue(np.greater_equal(predicted_disparity, 0.0))
        return predicted_depth, mixed_depth


def plot_complemented(zed_depth, predicted_depth, mixed_depth, cv_image, pngname=Path("full_depth.png")):
    vmin = -10
    vmax = -5.5
    h, w = cv_image.shape[:2]
    plt.figure(2, figsize=(16, 12))
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(-np.log(zed_depth), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK")
    plt.subplot(2, 2, 2)
    plt.imshow(-np.reshape(np.log(predicted_depth), (h, w)), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("depth anything")
    plt.subplot(2, 2, 3)
    if 1:
        print(f"{predicted_depth.shape=}")
        print(f"{zed_depth.shape=}")
    additional_depth = np.reshape(predicted_depth.copy(), (h, w))
    print(f"{additional_depth.shape=}")
    print(f"{zed_depth.shape=}")
    isfinite_pixels = np.isfinite(zed_depth)
    additional_depth[isfinite_pixels] = np.NAN
    plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.title("isnan")
    plt.subplot(2, 2, 4)
    plt.imshow(-np.log(mixed_depth), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK + depth anything")
    pngname.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(pngname)
    print(f"saved {pngname}")
