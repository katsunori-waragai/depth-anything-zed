"""
Sample script to perform depth completion
Input: zed-sdk camera
depends on:
    depth-anything:.
        Library to perform depth calculation for missing points (monocular depth calculation library)
Output:
    Depth data after completion processing
    Display in its pseudo-color (before and after completion)
"""

import pyzed.sl as sl
from pathlib import Path

import cv2
import numpy as np

from depanyzed import zed_camerainfo
from depanyzed import simpleply
import depanyzed


def main(quick: bool, save_depth: bool, save_ply: bool, save_fullply: bool):
    depth_engine = depanyzed.DepthEngine(
        frame_rate=30, raw=True, stream=True, record=False, save=False, grayscale=False
    )

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
    image_right = sl.Mat()
    depth = sl.Mat()
    depthimg = sl.Mat()
    point_cloud = sl.Mat()

    complementor = depanyzed.DepthComplementor()  # Model to complement depth in zed-sdk using depth-anything

    stable_max = None
    stable_min = None
    EPS = 1.0e-6

    cam_info = zed.get_camera_information()
    baseline = zed_camerainfo.get_baseline(cam_info)
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = zed_camerainfo.get_fx_fy_cx_cy(left_cam_params)
    print(f"{baseline=}")
    print(f"{fx=} {fy=} {cx=} {cy=}")
    input("hit any key to continue")
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)
            cv_image = image.get_data()
            cv_image = np.asarray(cv_image[:, :, :3])  # as RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depthの数値データ
            zed_depth = depth.get_data()  # np.ndarray 型
            zed.retrieve_image(depthimg, sl.VIEW.DEPTH)
            print(f"{zed_depth.shape=} {zed_depth.dtype=}")
            print(f"{np.nanpercentile(zed_depth, [5, 95])=}")

            da_disparity = depth_engine.infer_anysize(cv_image)
            assert da_disparity.shape[:2] == cv_image.shape[:2]

            isfinite_near = depanyzed.isfinite_near_pixels(zed_depth, da_disparity)
            if not complementor.predictable:
                real_disparity = depanyzed.depth_to_disparity(zed_depth)
                complementor.fit(da_disparity, real_disparity, isfinite_near)

            # logarithmic display of depth (completion process), logarithmic display of depth (depth_anything version)
            predicted_depth, mixed_depth = complementor.complement(zed_depth, da_disparity)
            assert predicted_depth.shape[:2] == da_disparity.shape[:2]

            use_direct_conversion = False
            if use_direct_conversion:
                depth_by_da = depanyzed.disparity_to_depth(disparity=da_disparity)
                assert depth_by_da.shape[:2] == da_disparity.shape[:2]
                predicted_depth = depth_by_da
            if save_depth:
                depth_file = Path("data/depth.npy")
                zed_depth_file = Path("data/zed_depth.npy")
                left_file = Path("data/left.png")
                depth_file.parent.mkdir(exist_ok=True, parents=True)
                np.save(depth_file, predicted_depth)
                np.save(zed_depth_file, zed_depth)
                cv2.imwrite(str(left_file), cv_image)
                print(f"saved {depth_file} {left_file}")

                cv_image_right = image_right.get_data()
                cv_image_right = np.asarray(cv_image_right[:, :, :3])  # as RGB
                right_file = Path("data/right.png")
                cv2.imwrite(str(right_file), cv_image_right)

            if save_ply:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                zed_ply_name = "data/pointcloud.ply"
                point_cloud.write(zed_ply_name)
                print(f"saved {zed_ply_name}")

            if save_fullply:
                depth2point = depanyzed.Depth2Points(fx, fy, cx, cy)
                points = depth2point.cloud_points(predicted_depth)
                H, W = predicted_depth.shape[:2]
                point_img = np.reshape(cv_image, (H * W, 3))
                selected_points = points[np.isfinite(predicted_depth.flatten())]
                selected_img = point_img[np.isfinite(predicted_depth.flatten())]
                full_plyname = "data/full_pointcloud.ply"
                simpleply.write_point_cloud(full_plyname, selected_points, selected_img)

                # Move the origin of the point cloud coordinates to ease display in meshlab.
                mean_point = np.mean(selected_points, axis=0)

                centered_points = selected_points.copy()
                centered_points[:, 0] -= mean_point[0]
                centered_points[:, 1] -= mean_point[1]
                centered_points[:, 2] -= mean_point[2]
                full_plyname2 = "data/full_pointcloud2.ply"
                simpleply.write_point_cloud(full_plyname2, centered_points, selected_img)
                print(f"saved {full_plyname2}")

            if not quick:
                full_depth_pngname = Path("data/full_depth.png")
                depanyzed.plot_complemented(zed_depth, predicted_depth, mixed_depth, cv_image, full_depth_pngname)
            else:
                log_zed_depth = np.log(zed_depth + EPS)
                assert log_zed_depth.shape == predicted_depth.shafiniteminpe
                assert log_zed_depth.dtype == predicted_depth.dtype
                concat_img = np.hstack((log_zed_depth, np.log(predicted_depth)))
                minval = depanyzed.finitemin(concat_img)
                maxval = depanyzed.finitemax(concat_img)
                stable_max = max((maxval, stable_max)) if stable_max else maxval
                stable_min = max((minval, stable_min)) if stable_min else minval

                print(f"{minval=} {maxval=} {stable_min=} {stable_max=}")
                if maxval > minval:
                    cv2.imshow("complemented", depanyzed.depth_as_colorimage(-concat_img))
                key = cv2.waitKey(1)

            i += 1

    zed.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scaled depth-anything")
    parser.add_argument("--quick", action="store_true", help="simple output without matplotlib")
    parser.add_argument("--save_depth", action="store_true", help="save depth and left image")
    parser.add_argument("--save_ply", action="store_true", help="save ply")
    parser.add_argument("--save_fullply", action="store_true", help="save full ply")
    args = parser.parse_args()
    main(args.quick, args.save_depth, args.save_ply, args.save_fullply)
