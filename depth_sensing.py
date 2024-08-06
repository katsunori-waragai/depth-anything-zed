import pyzed.sl as sl
import math
import numpy as np
import sys
import math

import cv2
import sklearn.linear_model
import matplotlib.pylab as plt

from lib_depth_engine import depth_as_colorimage
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

    # Create a Camera object
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
    point_cloud = sl.Mat()
    depth_image = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

    EPS = 1e-6

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            cv_image = cv_image[:, :, :3] # as RGB
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            depth_data = depth.get_data()  # cv_image 型
            print(f"{depth_data.shape=} {depth_data.dtype=}")
            effective_zed_depth = depth_data[np.isfinite(depth_data)]

            # assert np.alltrue(np.isfinite(depth_data))  # fails
            depth_data_color = depth_as_colorimage(depth_data)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            cv_depth_img = depth_image.get_data()
            cv2.imshow("cv_depth_img", cv_depth_img)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)


            # depth-anything からもdepthの推測値を得ること
            frame = cv2.resize(cv_image, (960, 540))
            print(f"{cv_image.shape=} {cv_image.dtype=}")
            print(f"{frame.shape=} {frame.dtype=}")
            assert frame.dtype == np.uint8
            disparity_raw = depth_engine.infer(frame)
            h, w = cv_image.shape[:2]
            disparity_raw = cv2.resize(disparity_raw, (w, h))
            disparity_color = depth_as_colorimage(disparity_raw)

            effective_inferred = disparity_raw[np.isfinite(depth_data)]
            uneffective_inferred = disparity_raw[np.logical_not(np.isfinite(depth_data))]

            print(f"{np.max(effective_zed_depth)=}")
            print(f"{np.max(effective_inferred)=}")
            print(f"{np.max(uneffective_inferred)=}")
            X = np.asarray(effective_inferred)  # disparity
            X2 = np.asarray(uneffective_inferred)
            Y = np.asarray(effective_zed_depth)  # depth
            Y_full = np.asarray(frame + EPS)
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
            predicted_Y_full = np.exp(predicted_logY_full)
            predicted_depth = predicted_Y_full
            predicted_depth = predicted_depth.reshape((h, w))
            plt.figure(1)
            plt.clf()
            print(f"{ransac.estimator_.coef_=}")
            plt.plot(logX, logY, ".")
            plt.plot(logX, predicted_logY, ".")
            plt.plot(logX2, predicted_logY2, ".")
            plt.xlabel("Depth-Anything disparity (log)")
            plt.ylabel("ZED SDK depth (log)")
            plt.grid(True)
            plt.savefig("depth_cmp_log.png")
            plt.figure(2, figsize=(16, 12))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(cv_depth_img)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            # plt.imshow(predicted_depth)
            plt.imshow(- predicted_logY_full.reshape(h, w))
            plt.colorbar()
            plt.savefig("full_depth.png")

            assert disparity_color.shape[:2] == cv_image.shape[:2]
            cv2.imshow("depth_anything_color", disparity_color)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit

            i += 1
           

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
