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
import math

import cv2
import sklearn.linear_model
import matplotlib.pylab as plt

from lib_depth_engine import DepthEngine

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
    depth_image = sl.Mat()

    EPS = 1e-6

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            cv_image = np.asarray(cv_image[:, :, :3]) # as RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            depth_data = depth.get_data()  # cv_image 型
            print(f"{depth_data.shape=} {depth_data.dtype=}")
            print(f"{np.nanpercentile(depth_data, [5, 95])=}")
            isfinite_pixels = np.isfinite(depth_data)
            isnear = np.less(depth_data, 1000)  # [mm]
            isfinite_near = np.logical_and(isfinite_pixels, isnear)
            effective_zed_depth = depth_data[isfinite_near]

            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            cv_depth_img = depth_image.get_data()

            # depth-anything からもdepthの推測値を得ること
            print(f"{cv_image.shape=} {cv_image.dtype=}")
            disparity_raw = depth_engine.infer_anysize(cv_image)
            print(f"{disparity_raw.shape=} {cv_image.shape=}")
            print(f"{disparity_raw.dtype=} {cv_image.dtype=}")
            assert disparity_raw.shape[:2] == cv_image.shape[:2]
            h, w = cv_image.shape[:2]

            effective_inferred = disparity_raw[isfinite_near]
            uneffective_inferred = disparity_raw[np.logical_not(isfinite_near)]

            print(f"{np.max(effective_zed_depth)=}")
            print(f"{np.max(effective_inferred)=}")
            print(f"{np.max(uneffective_inferred)=}")
            X = np.asarray(effective_inferred)  # disparity
            X2 = np.asarray(uneffective_inferred)
            Y = np.asarray(effective_zed_depth)  # depth
            assert np.alltrue(np.isfinite(X))
            assert np.alltrue(np.isfinite(Y))

            X_full = disparity_raw.flatten()

            logX = np.log(X + EPS)
            logX2 = np.log(X2 + EPS)
            logY = np.log(Y + EPS)

            logX_full = np.log(X_full + EPS)

            assert np.alltrue(np.isfinite(logX))
            assert np.alltrue(np.isfinite(logY))
            logX = logX.reshape(-1, 1)
            logX2 = logX2.reshape(-1, 1)
            logY = logY.reshape(-1, 1)
            logX_full = logX_full.reshape(-1, 1)

            print(f"{X.shape=} {X.dtype=}")
            print(f"{Y.shape=} {Y.dtype=}")

            ransac = sklearn.linear_model.RANSACRegressor()
            ransac.fit(logX, logY)
            predicted_logY = ransac.predict(logX)
            predicted_logY2 = ransac.predict(logX2)

            predicted_logY_full = ransac.predict(logX_full)
            predicted_logY_full2 = np.reshape(predicted_logY_full.copy(), (h, w))
            predicted_logY_full2[isfinite_near] = np.log(depth_data)[isfinite_near]
            plt.figure(1)
            plt.clf()
            print(f"{ransac.estimator_.coef_=}")
            plt.plot(logX, logY, ".")
            plt.plot(logX, predicted_logY, ".")
#            plt.plot(logX2, predicted_logY2, ".")
            plt.xlabel("Depth-Anything disparity (log)")
            plt.ylabel("ZED SDK depth (log)")
            plt.grid(True)
            plt.savefig("depth_cmp_log.png")

            predicted_logY_full = predicted_logY_full.reshape(h, w)
            vmin = -10
            vmax = -5.5
            plt.figure(2, figsize=(16, 12))
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(- np.log(depth_data), vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title("ZED SDK")
            plt.subplot(2, 2, 2)
            plt.imshow(- predicted_logY_full, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title("depth anything")
            plt.subplot(2, 2, 3)
            assert predicted_logY_full.shape[:2] == depth_data.shape[:2]
            additional_depth = predicted_logY_full.copy()
            additional_depth[isfinite_pixels] = np.NAN
            plt.imshow(cv_image)
            plt.colorbar()
            plt.title("isnan")
            plt.subplot(2, 2, 4)
            plt.imshow(- predicted_logY_full2, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title("ZED SDK + depth anything")
            plt.savefig("full_depth.png")

            i += 1
           

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
