"""
This is main script using lib_depth_engine.py without zed-sdk
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np

import pycuda.autoinit  # don't remove. Otherwise import pycuda.autoinit

from depanyzed import simpleply
import depanyzed

PROJECT_PATH = Path(__file__).resolve().parent
WEIGHT_DIR = PROJECT_PATH / "weights"

assert (WEIGHT_DIR / "depth_anything_vits14_308.trt").is_file()


def depth_for_usb(args):
    """
    depth-anything for ZED2i as USB camera
    """
    depth_engine = depanyzed.DepthEngine(
        frame_rate=args.frame_rate,
        raw=True,
        stream=True,
        record=False,
        save=False,
        grayscale=False,
        trt_engine_path=WEIGHT_DIR / "depth_anything_vits14_308.trt",
    )
    save_ply = False
    cap = cv2.VideoCapture(0)
    while True:
        _, orig_frame = cap.read()
        # stereo camera left part
        H_, w_ = orig_frame.shape[:2]
        orig_frame = orig_frame[:, : w_ // 2, :]
        original_height, original_width = orig_frame.shape[:2]
        frame = cv2.resize(orig_frame, (960, 540))
        print(f"{frame.shape=} {frame.dtype=}")
        depth_raw = depth_engine.infer(frame)

        depth = depanyzed.depth_as_colorimage(depth_raw)
        results = np.concatenate((frame, depth), axis=1)

        depth_raw_orignal_size = cv2.resize(
            depth_raw, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        )
        if save_ply:
            points = depanyzed.to_point_cloud_np(depth_raw_orignal_size)
            plyname = Path("tmp.ply")
            simpleply.write_point_cloud(plyname, points, orig_frame)
            print(f"saved {plyname}")

        if depth_engine.record:
            depth_engine.video.write(results)

        if depth_engine.save:
            cv2.imwrite(str(depth.save_path / f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png'), results)

        if depth_engine.stream:
            cv2.imshow("Depth", results)  # This causes bad performance

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                depth_raw_orignal_size = cv2.resize(
                    depth_raw, (original_width, original_height), interpolation=cv2.INTER_NEAREST
                )
                points = depanyzed.to_point_cloud_np(depth_raw_orignal_size)

                plyname = Path("tmp.ply")
                simpleply.write_point_cloud(plyname, points, orig_frame)
                print(f"saved {plyname}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="depth-anything using zed2i as usb camera")
    args.add_argument("--frame_rate", type=int, default=15, help="Frame rate of the camera")
    args.add_argument("--raw", action="store_true", help="Use only the raw depth map")
    args.add_argument("--stream", action="store_true", help="Stream the results")
    args.add_argument("--record", action="store_true", help="Record the results")
    args.add_argument("--save", action="store_true", help="Save the results")
    args.add_argument("--grayscale", action="store_true", help="Convert the depth map to grayscale")
    args = args.parse_args()

    depth_for_usb(args)
