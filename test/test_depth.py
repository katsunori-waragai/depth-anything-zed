import cv2
import numpy as np

from pathlib import Path
import datetime

from depanyzed.lib_depth_engine import DepthEngine, depth_as_colorimage

test_module_path = Path(__file__).resolve().parent
weights_dir = test_module_path.parent / "weights"

print(f"{weights_dir=}")

def depth_run(args):
    print(f"{args=}")
    depth_engine = DepthEngine(
        frame_rate=args.frame_rate, raw=True, stream=True, record=False, save=True, grayscale=False,
        trt_engine_path=weights_dir / "depth_anything_vits14_308.trt"
    )
    save_ply = False
    cap = cv2.VideoCapture(0)
    imgname = Path("./data/left.png")
    orig_frame = cv2.imread(str(imgname))
    _, orig_frame = cap.read()
    # stereo camera left part
    H_, w_ = orig_frame.shape[:2]
    orig_frame = orig_frame[:, : w_ // 2, :]
    original_height, original_width = orig_frame.shape[:2]
    frame = cv2.resize(orig_frame, (960, 540))
    print(f"{frame.shape=} {frame.dtype=}")
    depth_raw = depth_engine.infer(frame)

    depth_img = depth_as_colorimage(depth_raw)
    results = np.concatenate((frame, depth_img), axis=1)

    depth_raw_orignal_size = cv2.resize(
        depth_raw, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    assert depth_raw_orignal_size.shape[:2] == orig_frame.shape[:2]
    assert len(depth_raw_orignal_size.shape) == 2
    assert depth_raw_orignal_size.dtype in (np.float32, np.float64)

    if depth_engine.save:
        cv2.imwrite(str(depth_engine.save_path / f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png'), results)
        print(f"{depth_engine.save_path=}")

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="depth-anything using zed2i as usb camera")
    args.add_argument("--frame_rate", type=int, default=15, help="Frame rate of the camera")
    args.add_argument("--raw", action="store_true", help="Use only the raw depth map")
    args.add_argument("--stream", action="store_true", help="Stream the results")
    args.add_argument("--record", action="store_true", help="Record the results")
    args.add_argument("--save", action="store_true", help="Save the results")
    args.add_argument("--grayscale", action="store_true", help="Convert the depth map to grayscale")
    args = args.parse_args()
    depth_run(args)