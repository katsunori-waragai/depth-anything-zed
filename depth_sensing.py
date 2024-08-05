import pyzed.sl as sl
import math
import numpy as np
import sys
import math

import cv2


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
            depth_data_color = depth_as_colorimage(depth_data)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            cv_depth_img = depth_image.get_data()
            cv2.imshow("cv_depth_img", cv_depth_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)


            # depth-anything からもdepthの推測値を得ること
            frame = cv2.resize(cv_image, (960, 540))
            print(f"{cv_image.shape=} {cv_image.dtype=}")
            print(f"{frame.shape=} {frame.dtype=}")
            assert frame.dtype == np.uint8
            depth_raw = depth_engine.infer(frame)

            depth_color = depth_as_colorimage(depth_raw)
            h, w = cv_image.shape[:2]
            depth_color = cv2.resize(depth_color, (w, h))

            assert depth_color.shape[:2] == image.shape[:2]

            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            if math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])
                print(f"Distance to Camera at {{{x};{y}}}: {distance}")
            else : 
                print(f"The distance can not be computed at {{{x};{y}}}")
            i += 1    
           

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
