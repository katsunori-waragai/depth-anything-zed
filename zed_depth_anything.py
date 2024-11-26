"""
Sample script to view depth images
- ZED-SDK depth
- Depth-Anything depth(native)
This is developing code for depth-anything with zed sdk.
"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

import depanyzed

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 2.0  # [m]


def parse_args(opt, init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo2"):
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


def as_3channel(cv_image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv_image[:, :, :3])


def main(opt):
    depth_engine = depanyzed.DepthEngine(
        frame_rate=15, raw=True, stream=True, record=False, save=False, grayscale=False
    )

    zed = sl.Camera()
    init_params = sl.InitParameters()
    parse_args(opt, init_params)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD2K

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    image = sl.Mat()
    right_image = sl.Mat()
    depth = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")

    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
            cv_image = image.get_data()
            zed.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU)
            cv_right_image = right_image.get_data()
            assert cv_image.shape[2] == 4  # ZED SDK dependent.

            cv_image = as_3channel(cv_image)
            cv_right_image = as_3channel(cv_right_image)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            cv_depth_data = depth.get_data()
        else:
            continue
        assert cv_image.shape[2] == 3
        assert cv_image.dtype == np.uint8
        frame = cv2.resize(cv_image, (960, 540))
        cv_right_image = cv2.resize(cv_right_image, (960, 540))
        assert frame.shape[0] == 540
        assert frame.shape[1] == 960
        depth_any_raw = depth_engine.infer(frame)
        depth_any = depanyzed.depth_as_colorimage(depth_any_raw)
        assert frame.dtype == depth_any.dtype
        assert frame.shape[0] == depth_any.shape[0]
        depth_colored = depanyzed.depth_as_colorimage(np.reciprocal(cv_depth_data))
        depth_colored = cv2.resize(depth_colored, (960, 540))
        print(f"{depth_colored.shape=} {frame.shape[:2]=}")
        assert depth_colored.shape[:2] == frame.shape[:2]
        upper =np.concatenate((frame, cv_right_image), axis=1)
        lower = np.concatenate((depth_colored, depth_any), axis=1)
        results = np.concatenate((upper, lower), axis=0)
        cv2.imshow("Depth", results)
        cv2.waitKey(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depth-anything(native) with zed camera")
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
        default=90,
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
