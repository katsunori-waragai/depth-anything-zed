from pathlib import Path

import cv2
import numpy as np

from depanyzed.lib_depth_engine import DepthEngine, depth_as_colorimage

test_module_path = Path(__file__).resolve().parent
weights_dir = test_module_path.parent / "weights"


def test_depth_run():
    import argparse

    args = argparse.Namespace(frame_rate=15, grayscale=False, raw=False, record=False, save=False, stream=False)
    print(f"{args=}")
    depth_engine = DepthEngine(
        frame_rate=args.frame_rate,
        raw=True,
        stream=True,
        record=False,
        save=True,
        grayscale=False,
        trt_engine_path=weights_dir / "depth_anything_vits14_308.trt",
    )
    imgname = Path("./data/left.png")
    orig_frame = cv2.imread(str(imgname))
    # stereo camera left part
    H_, w_ = orig_frame.shape[:2]
    orig_frame = orig_frame[:, : w_ // 2, :]
    original_height, original_width = orig_frame.shape[:2]
    frame = cv2.resize(orig_frame, (960, 540))
    print(f"{frame.shape=} {frame.dtype=}")
    depth_raw = depth_engine.infer(frame)

    depth_img = depth_as_colorimage(depth_raw)
    result_image = np.concatenate((frame, depth_img), axis=1)

    depth_raw_orignal_size = cv2.resize(depth_raw, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    assert depth_raw_orignal_size.shape[:2] == orig_frame.shape[:2]
    assert len(depth_raw_orignal_size.shape) == 2
    assert depth_raw_orignal_size.dtype in (np.float32, np.float64)

    cv2.imwrite("test_result.png", result_image)
