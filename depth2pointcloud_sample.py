"""
Script to create point cloud data file (*.ply) based on npy file and left image of depth
"""

import numpy as np
import cv2

from depanyzed.depth2pointcloud import Depth2Points
from depanyzed import simpleply

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("depth npy to point cloud")
    parser.add_argument("--depth", default="test/data/zed_depth.npy", help="depth npy file")
    parser.add_argument("--rgb", default="test/data/left.png", help="left rgb image file")
    args = parser.parse_args()

    depth_file = args.depth
    rgb_file = args.rgb
    depth = np.load(depth_file)
    img = cv2.imread(rgb_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # [LEFT_CAM_2K]
    # fx = 1064.82
    # fy = 1065.07
    # cx = 1099.05
    # cy = 628.813

    if img.shape[1] == 1280:
        # [LEFT_CAM_HD]
        fx = 532.41
        fy = 532.535
        cx = 636.025  # [pixel]
        cy = 362.4065  # [pixel]
    else:
        print(f"need setting for {image.shape}")
        exit

    H, W = depth.shape[:2]

    depth2point = Depth2Points(fx, fy, cx, cy)
    points = depth2point.cloud_points(depth)

    assert depth.shape[:2] == img.shape[:2]
    point_img = np.reshape(img, (H * W, 3))
    selected_points = points[np.isfinite(depth.flatten())]
    selected_img = point_img[np.isfinite(depth.flatten())]
    print(f"{points.shape=}")
    plyname = "data/test.ply"

    # Move the origin of the point cloud coordinates to ease display in meshlab.
    mean_point = np.mean(selected_points, axis=0)

    centered_points = selected_points.copy()
    centered_points[:, 0] -= mean_point[0]
    centered_points[:, 1] -= mean_point[1]
    centered_points[:, 2] -= mean_point[2]
    plyname2 = "data/test_centered.ply"

    simpleply.write_point_cloud(plyname, selected_points, selected_img)
    simpleply.write_point_cloud(plyname2, centered_points, selected_img)
