from pathlib import Path

import cv2
import numpy as np

from capture import depth_as_colorimage, depth_as_gray


def main(args):
    captured_dir = Path(args.captured_dir)
    leftdir = captured_dir / "left"
    rightdir = captured_dir / "right"
    zeddepthdir = captured_dir / "zed-depth"

    left_images = sorted(leftdir.glob("*.png"))
    depth_npys = sorted(zeddepthdir.glob("*.npy"))
    for leftname, depth_name in zip(left_images, depth_npys):
        print(leftname, depth_name)
        image = cv2.imread(leftname)
        depth = np.load(depth_name)

        colored_depth = depth_as_colorimage(depth) if args.jet else depth_as_gray(depth)
        assert image.shape == colored_depth.shape
        assert image.dtype == colored_depth.dtype
        results = np.concatenate((image, colored_depth), axis=1)
        cv2.imsow(results)
        cv2.waitKey(10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="depth npy file viewer")
    parser.add_argument("captured_dir", help="captured directory by capture.py")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    args = parser.parse_args()
    print(args)
