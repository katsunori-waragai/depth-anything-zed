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

import pyzed.sl as sl
import math
import numpy as np
import sys
import time
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import sklearn.linear_model
import matplotlib.pylab as plt

from lib_depth_engine import DepthEngine, depth_as_colorimage, finitemin, finitemax


def isfinite_near_pixels(zed_depth: np.ndarray, da_disparity: np.ndarray, far_depth_limit=1000, small_disparity_limit=math.exp(0.5)):
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

    """

    ransac = sklearn.linear_model.RANSACRegressor()
    EPS = 1e-6
    predictable = False  # 最初のフィッティングがされないうちは、predict()できない。

    def fit(self, zed_depth: np.ndarray, da_disparity: np.ndarray, isfinite_near: np.ndarray):
        """
        isfinite_near がtrue である画素について、zed sdkのdepthと depth anything の視差(disparity) との関係式を算出する。
        - RANSAC のアルゴリズムを使用。
        - 深度は視差に反比例します。
        - そのため、log(深度) とlog(視差) は-1の勾配で1次式で表されると期待します。
        """
        t0 = cv2.getTickCount()
        assert zed_depth.shape[:2] == da_disparity.shape[:2]
        effective_zed_depth = zed_depth[isfinite_near]
        effective_inferred = da_disparity[isfinite_near]

        print(f"{np.max(effective_zed_depth)=}")
        print(f"{np.max(effective_inferred)=}")
        X = np.asarray(effective_inferred)  # disparity
        Y = np.asarray(effective_zed_depth)  # depth
        logX = np.log(X + self.EPS)
        logY = np.log(Y + self.EPS)
        logX = logX.reshape(-1, 1)
        logY = logY.reshape(-1, 1)
        print(f"{X.shape=} {X.dtype=}")
        print(f"{Y.shape=} {Y.dtype=}")

        self.ransac.fit(logX, logY)
        self.predictable = True
        inlier_mask = self.ransac.inlier_mask_
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in fit")
        predicted_logY = self.predict_log(logX)
        if True:
            self.regression_plot(logX, logY, predicted_logY, inlier_mask)

    def predict_log(self, logX: np.ndarray) -> np.ndarray:
        """
        zed_depthでの欠損値を、depth-anything 由来の値で補完して返す。

        returns log_depth
        """
        t0 = cv2.getTickCount()
        assert self.predictable
        r = self.ransac.predict(logX)
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in predict_log")
        return r

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        zed_depthでの欠損値を、depth-anything 由来の値で補完して返す。

        returns log_depth
        """
        t0 = cv2.getTickCount()
        assert self.predictable
        r = self.predict_log(np.log(X + self.EPS))
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in predict")
        return np.exp(r)

    def regression_plot(self, logX: np.ndarray, logY: np.ndarray, predicted_logY: np.ndarray, inlier_mask):
        plt.figure(1, figsize=(8, 6))
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(logX, logY, ".")
        plt.plot(logX, predicted_logY, ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity (log)")
        plt.ylabel("ZED SDK depth (log)")
        plt.grid(True)
        plt.subplot(2, 2, 2)

        plt.plot(logX[inlier_mask], logY[inlier_mask], ".")
        plt.plot(logX, predicted_logY, ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity (log)")
        plt.ylabel("ZED SDK depth (log)")
        plt.grid(True)
        plt.subplot(2, 2, 4)

        plt.plot(logX[inlier_mask], logY[inlier_mask] - predicted_logY[inlier_mask], ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity (log)")
        plt.ylabel("ZED SDK depth/predicted_depth (log)")
        plt.grid(True)
        plt.savefig("depth_cmp_log.png")

    def complement(self, zed_depth: np.ndarray, da_disparity: np.ndarray):
        """
        input, output in linear scale
        """
        h, w = zed_depth.shape[:2]
        X_full = da_disparity.flatten().reshape(-1, 1)
        predicted_depth = self.predict(X_full)
        predicted_depth2 = np.reshape(predicted_depth.copy(), (h, w))
        isfinite_near = isfinite_near_pixels(zed_depth, da_disparity)
        predicted_depth2[isfinite_near] = zed_depth[isfinite_near]
        return predicted_depth2, predicted_depth


def plot_complemented(zed_depth, predicted_log_depth, predicted_log_depth2, cv_image):
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
    plt.imshow(-np.reshape(predicted_log_depth, (h, w)), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("depth anything")
    plt.subplot(2, 2, 3)
    if 1:
        print(f"{predicted_log_depth.shape=}")
        print(f"{zed_depth.shape=}")
    additional_depth = np.reshape(predicted_log_depth.copy(), (h, w))
    print(f"{additional_depth.shape=}")
    print(f"{zed_depth.shape=}")
    isfinite_pixels = np.isfinite(zed_depth)
    additional_depth[isfinite_pixels] = np.NAN
    plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.title("isnan")
    plt.subplot(2, 2, 4)
    plt.imshow(-predicted_log_depth2, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK + depth anything")
    pngname = "full_depth.png"
    plt.savefig(pngname)
    print(f"saved {pngname}")


def main(quick: bool, save_depth: bool, save_ply: bool):
    # depth_anything の準備をする。
    depth_engine = DepthEngine(frame_rate=30, raw=True, stream=True, record=False, save=False, grayscale=False)

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    depthimg = sl.Mat()
    point_cloud = sl.Mat()

    complementor = DepthComplementor()  # depth-anythingを使ってzed-sdk でのdepthを補完するモデル

    stable_max = None
    stable_min = None
    EPS = 1.0e-6

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            cv_image = np.asarray(cv_image[:, :, :3])  # as RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            zed_depth = depth.get_data()  # np.ndarray 型
            zed.retrieve_image(depthimg, sl.VIEW.DEPTH)
            print(f"{zed_depth.shape=} {zed_depth.dtype=}")
            print(f"{np.nanpercentile(zed_depth, [5, 95])=}")

            # depth-anything からもdepthの推測値を得る
            da_disparity = depth_engine.infer_anysize(cv_image)
            assert da_disparity.shape[:2] == cv_image.shape[:2]

            isfinite_near = isfinite_near_pixels(zed_depth, da_disparity)
            if not complementor.predictable:
                complementor.fit(zed_depth, da_disparity, isfinite_near)

            # 対数表示のdepth（補完処理）、対数表示のdepth(depth_anything版）
            predicted_depth2, predicted_depth = complementor.complement(zed_depth, da_disparity)
            if save_depth:
                depth_file = Path("data/depth.npy")
                zed_depth_file = Path("data/zed_depth.npy")
                left_file = Path("data/left.png")
                depth_file.parent.mkdir(exist_ok=True, parents=True)
                np.save(depth_file, predicted_depth)
                np.save(zed_depth_file, zed_depth)
                cv2.imwrite(str(left_file), cv_image)
                print(f"saved {depth_file} {left_file}")

            if save_ply:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                zed_ply_name = "data/pointcloud.ply"
                point_cloud.write(zed_ply_name)
                print(f"saved {zed_ply_name}")

            if not quick:
                plot_complemented(zed_depth, predicted_depth, predicted_depth2, cv_image)
                time.sleep(5)
            else:
                log_zed_depth = np.log(zed_depth + EPS)
                concat_img = np.hstack((log_zed_depth, np.log(predicted_depth2)))
                minval = finitemin(concat_img)
                maxval = finitemax(concat_img)
                stable_max = max((maxval, stable_max)) if stable_max else maxval
                stable_min = max((minval, stable_min)) if stable_min else minval

                print(f"{minval=} {maxval=} {stable_min=} {stable_max=}")
                if maxval > minval:
                    cv2.imshow("complemented", depth_as_colorimage(-concat_img))
                key = cv2.waitKey(1)

            i += 1

    zed.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("depth sensing")
    parser.add_argument("--quick", action="store_true", help="simple output without matplotlib")
    parser.add_argument("--save_depth", action="store_true", help="save depth and left image")
    parser.add_argument("--save_ply", action="store_true", help="save ply")
    args = parser.parse_args()
    main(args.quick, args.save_depth, args.save_ply)
