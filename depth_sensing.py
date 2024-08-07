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

from lib_depth_engine import DepthEngine


def isfinite_near_pixels(depth_data, disparity_raw):
    isfinite_pixels = np.isfinite(depth_data)
    isnear = np.less(depth_data, 1000)  # [mm]
    isnear_da = np.greater(disparity_raw, math.exp(0.5))
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

    def fit(self, depth_data, disparity_raw, isfinite_near):
        t0 = cv2.getTickCount()
        assert depth_data.shape[:2] == disparity_raw.shape[:2]
        effective_zed_depth = depth_data[isfinite_near]
        effective_inferred = disparity_raw[isfinite_near]

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

    def complement(self, depth_data, disparity_raw):
        h, w = depth_data.shape[:2]
        X_full = disparity_raw.flatten()
        logX_full = np.log(X_full + self.EPS)
        logX_full = logX_full.reshape(-1, 1)

        predicted_logY_full = self.predict(logX_full)
        predicted_logY_full2 = np.reshape(predicted_logY_full.copy(), (h, w))
        isfinite_near = isfinite_near_pixels(depth_data, disparity_raw)
        predicted_logY_full2[isfinite_near] = np.log(depth_data)[isfinite_near]
        return predicted_logY_full2, predicted_logY_full


def plot_complemented(depth_data, predicted_logY_full, predicted_logY_full2, cv_image):
    vmin = -10
    vmax = -5.5
    h, w = cv_image.shape[:2]
    plt.figure(2, figsize=(16, 12))
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(- np.log(depth_data), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK")
    plt.subplot(2, 2, 2)
    plt.imshow(- np.reshape(predicted_logY_full, (h, w)), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("depth anything")
    plt.subplot(2, 2, 3)
    if 1:
        print(f"{predicted_logY_full.shape=}")
        print(f"{depth_data.shape=}")
    additional_depth = np.reshape(predicted_logY_full.copy(), (h, w))
    print(f"{additional_depth.shape=}")
    print(f"{depth_data.shape=}")
    isfinite_pixels = np.isfinite(depth_data)
    additional_depth[isfinite_pixels] = np.NAN
    plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.title("isnan")
    plt.subplot(2, 2, 4)
    plt.imshow(- predicted_logY_full2, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("ZED SDK + depth anything")
    pngname = "full_depth.png"
    plt.savefig(pngname)
    print(f"saved {pngname}")


def main():
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

    complementor = DepthComplementor()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            cv_image = np.asarray(cv_image[:, :, :3]) # as RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            depth_data = depth.get_data()  # cv_image 型
            print(f"{depth_data.shape=} {depth_data.dtype=}")
            print(f"{np.nanpercentile(depth_data, [5, 95])=}")

            # depth-anything からもdepthの推測値を得る
            disparity_raw = depth_engine.infer_anysize(cv_image)
            assert disparity_raw.shape[:2] == cv_image.shape[:2]

            isfinite_near = isfinite_near_pixels(depth_data, disparity_raw)

            complementor.fit(depth_data, disparity_raw, isfinite_near)
            h, w = cv_image.shape[:2]
            predicted_logY_full2, predicted_logY_full = complementor.complement(depth_data, disparity_raw)

            plot_complemented(depth_data, predicted_logY_full, predicted_logY_full2, cv_image)
            time.sleep(5)

            i += 1
           

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
