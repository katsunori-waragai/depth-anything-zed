"""
Depth Anything を用いて欠損点のdepthを補完する。

TODO：
- RANSACでの合わせ込みの際に、遠方側の点は除外しよう。
    - 遠方側で関係式を合わせようとすると誤差が大きくなるはずのため。
    - 視差とdepthとの関係式は depth ∝ 1/視差　であるべき。
    - log-log プロットで傾き　-1であるべき。
    - 遠方側は除外して、近距離用のRANSACでフィッティングを作ろう。
- logスケールでのfittingの残差を表示すること。
    - 残差の分だけ距離を間違えることになる。
- 補完後のdepthから、3DのpointCloud を得られるようにすること。

-
"""

import pyzed.sl as sl
import math
import numpy as np
import sys
import time
import math
from dataclasses import dataclass

import cv2
import sklearn.linear_model
import matplotlib.pylab as plt

from lib_depth_engine import DepthEngine, depth_as_colorimage


def isfinite_near_pixels(zed_depth, ad_disparity):
    """
    RANSAC で合わせこみをする際に、事前に選択する画素をboolの配列として選択する。
    """
    isfinite_pixels = np.isfinite(zed_depth)
    isnear = np.less(zed_depth, 1000)  # [mm]
    isnear_da = np.greater(ad_disparity, math.exp(0.5))
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
    predictable = False

    def fit(self, zed_depth, ad_disparity, isfinite_near):
        t0 = cv2.getTickCount()
        assert zed_depth.shape[:2] == predicted_log_depthdisparity_raw.shape[:2]
        effective_zed_depth = zed_depth[isfinite_near]
        effective_inferred = ad_disparity[isfinite_near]

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
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in fit")
        if True:
            predicted_logY = self.predict(logX)
            self.regression_plot(logX, logY, predicted_logY)

    def predict(self, logX):
        """
        returns log_depth
        """
        t0 = cv2.getTickCount()
        assert self.predictable
        r = self.ransac.predict(logX)
        t1 = cv2.getTickCount()
        used = (t1 - t0) / cv2.getTickFrequency()
        print(f"{used} [s] in predict")
        return r

    def regression_plot(self, logX, logY, predicted_logY):
        plt.figure(1)
        plt.clf()
        plt.plot(logX, logY, ".")
        plt.plot(logX, predicted_logY, ".")
        #            plt.plot(logX2, predicted_logY2, ".")
        plt.xlabel("Depth-Anything disparity (log)")
        plt.ylabel("ZED SDK depth (log)")
        plt.grid(True)
        plt.savefig("depth_cmp_log.png")

    def complement(self, zed_depth, ad_disparity):
        h, w = zed_depth.shape[:2]
        X_full = ad_disparity.flatten()
        logX_full = np.log(X_full + self.EPS)
        logX_full = logX_full.reshape(-1, 1)

        predicted_log_depth = self.predict(logX_full)
        predicted_log_depth2 = np.reshape(predicted_log_depth.copy(), (h, w))
        isfinite_near = isfinite_near_pixels(zed_depth, ad_disparity)
        predicted_log_depth2[isfinite_near] = np.log(zed_depth)[isfinite_near]
        return predicted_log_depth2, predicted_log_depth


def plot_complemented(zed_depth, predicted_log_depth, predicted_log_depth2, cv_image):
    vmin = -10
    vmax = -5.5
    h, w = cv_image.shape[:2]
    plt.figure(2, figsize=(16, 12))
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(- np.log(zed_depth), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK")
    plt.subplot(2, 2, 2)
    plt.imshow(- np.reshape(predicted_log_depth, (h, w)), vmin=vmin, vmax=vmax)
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
    plt.imshow(- predicted_log_depth2, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK + depth anything")
    pngname = "full_depth.png"
    plt.savefig(pngname)
    print(f"saved {pngname}")


def main(quick: bool):
    # depth_anything の準備をする。
    depth_engine = DepthEngine(
        frame_rate=30,
        raw=True,
        stream=True,
        record=False,
        save=False,
        grayscale=False
    )

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    depthimg = sl.Mat()

    complementor = DepthComplementor()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            cv_image = np.asarray(cv_image[:, :, :3]) # as RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            zed_depth = depth.get_data()  # np.ndarray 型
            zed.retrieve_image(depthimg, sl.VIEW.DEPTH)
            print(f"{zed_depth.shape=} {zed_depth.dtype=}")
            print(f"{np.nanpercentile(zed_depth, [5, 95])=}")

            # depth-anything からもdepthの推測値を得る
            ad_disparity = depth_engine.infer_anysize(cv_image)
            assert ad_disparity.shape[:2] == cv_image.shape[:2]

            isfinite_near = isfinite_near_pixels(zed_depth, ad_disparity)
            if not complementor.predictable:
                complementor.fit(zed_depth, ad_disparity, isfinite_near)
            h, w = cv_image.shape[:2]
            predicted_log_depth2, predicted_log_depth = complementor.complement(zed_depth, ad_disparity)

            if not quick:
                plot_complemented(zed_depth, predicted_log_depth, predicted_log_depth2, cv_image)
                time.sleep(5)
            else:
                depth_mono_image = depthimg.get_data()
                cv2.imshow("zed", depth_as_colorimage(depth_mono_image[:, :, 0]))
                # cv2.imshow("zed", depth_as_colorimage(- np.log(np.abs(zed_depth))))
                key = cv2.waitKey(1)
                cv2.imshow("complemented", depth_as_colorimage(- predicted_log_depth2))
                key = cv2.waitKey(1)

            i += 1
           

    # Close the camera
    zed.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("depth sensing")
    parser.add_argument("--quick", action="store_true", help="simple output without matplotlib")
    args = parser.parse_args()
    main(args.quick)
