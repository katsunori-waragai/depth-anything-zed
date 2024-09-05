from pathlib import Path
import time

import cv2
import numpy as np

from capture import depth_as_colorimage, depth_as_gray


def main(args):
    captured_dir = Path(args.captured_dir)
    leftdir = captured_dir / "left"
    rightdir = captured_dir / "right"
    zeddepthdir = captured_dir / "zed-depth"
    sec = args.sec

    left_images = sorted(leftdir.glob("*.png"))
    depth_npys = sorted(zeddepthdir.glob("**/*.npy"))
    for leftname, depth_name in zip(left_images, depth_npys):
        print(leftname, depth_name)
        image = cv2.imread(str(leftname))
        depth = np.load(str(depth_name))

        if args.gray:
            colored_depth = depth_as_gray(depth)
        elif args.jet:
            colored_depth = depth_as_colorimage(depth, colormap=cv2.COLORMAP_JET)
        elif args.inferno:
            colored_depth = depth_as_colorimage(depth, colormap=cv2.COLORMAP_INFERNO)
        else:
            colored_depth = depth_as_colorimage(depth, colormap=cv2.COLORMAP_JET)

        assert image.shape == colored_depth.shape
        assert image.dtype == colored_depth.dtype
        results = np.concatenate((image, colored_depth), axis=1)
        cv2.imshow("left depth", results)
        cv2.waitKey(10)
        time.sleep(sec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="depth npy file viewer")
    parser.add_argument("captured_dir", help="captured directory by capture.py")
    parser.add_argument("--sec", default=3, help="wait sec")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")
    args = parser.parse_args()
    print(args)
    main(args)
