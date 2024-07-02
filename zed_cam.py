"""
depth画像を見るサンプルスクリプト

"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

import matplotlib.pyplot as plt


import inspect

import lib_depth_engine

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 2.0  # [m]

def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))


def as_matrix(chw_array):
    H_, W_ = chw_array.shape[-2:]
    return np.reshape(chw_array, (H_, W_))


def main(opt):
    depth_engine = lib_depth_engine.DepthEngine(
        frame_rate=15,
        raw=False,
        stream=True,
        record=False,
        save=False,
        grayscale=False
    )


    prompt = "bottle . person . box"
    prompt = "bottle"
    watching_obj = "bottle"
    assert prompt.find(watching_obj) > -1

    zed = sl.Camera()

    init_params = sl.InitParameters()

    parse_args(init_params)

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    image = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    # runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")
    condition_str = f"mode: {init_params.depth_mode} conf: {runtime_parameters.confidence_threshold}"

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
            cv_image = image.get_data()
            cv_image = cv_image[:, :, :3].copy()
            assert cv_image.shape[2] == 3
            frame = cv2.resize(cv_image, (960, 540))
            depth_any = depth_engine.infer(frame)
            results = np.concatenate((frame, depth_any), axis=1)
            cv2.imshow('Depth', results)

            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth
            depth_map_data = depth_map.get_data()

            # 空間座標を得ることが必要。
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            points = point_cloud.get_data()
            print(f"{points.shape=}")

            # 点群の色情報が有効な領域をvalid_points_maskとして取得する。
            points_color = points[:, :, 3]
            valid_points_mask = np.isfinite(points_color)
            print(f"{valid_points_mask.shape=} {valid_points_mask.dtype=}")
            # points[y, x]で、元画像上の点と対応がつくのかどうか？

            depth_map_data_modified = depth_map_data.copy()
            print(f"{depth_map_data_modified.shape=} {depth_map_data_modified.dtype=}")
            depth_map_data_modified[np.logical_not(valid_points_mask)] = np.nan

    depth_map.free(memory_type=sl.MEM.CPU)
    point_cloud.free(memory_type=sl.MEM.CPU)
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="depth confidence_threshold(0 ~ 100)",
        default=100,
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
