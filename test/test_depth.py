import cv2
import numpy as np

from ptahlib import Path
import datetime

from depanyzed.lib_depth_engine import DepthEngine, depth_as_colorimage

def depth_run(args):
    depth_engine = DepthEngine(
        frame_rate=args.frame_rate, raw=True, stream=True, record=False, save=False, grayscale=False
    )
    save_ply = False
    cap = cv2.VideoCapture(0)
    imgname = Path("./data/left.png")
    orig_frame = cv2.imread(imgname)
    _, orig_frame = cap.read()
    # stereo camera left part
    H_, w_ = orig_frame.shape[:2]
    orig_frame = orig_frame[:, : w_ // 2, :]
    original_height, original_width = orig_frame.shape[:2]
    frame = cv2.resize(orig_frame, (960, 540))
    print(f"{frame.shape=} {frame.dtype=}")
    depth_raw = depth_engine.infer(frame)

    depth = depth_as_colorimage(depth_raw)
    results = np.concatenate((frame, depth), axis=1)

    depth_raw_orignal_size = cv2.resize(
        depth_raw, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    if depth_engine.save:
        cv2.imwrite(str(depth.save_path / f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png'), results)
