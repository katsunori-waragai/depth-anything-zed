from pathlib import Path

import cv2
import numpy as np

from depanyzed.depth2pointcloud import Depth2Points
from depanyzed import simpleply


def test_Depth2Points():

    depth_file = "../test/data/zed_depth.npy"
    rgb_file = "../test/data/left.png"
    depth = np.load(depth_file)
    img = cv2.imread(rgb_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    assert img.shape[1] == 1280
    if img.shape[1] == 1280:
        # [LEFT_CAM_HD]
        fx = 532.41
        fy = 532.535
        cx = 636.025  # [pixel]
        cy = 362.4065  # [pixel]
    else:
        print(f"need setting for {img.shape}")
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

    assert points.shape[1] == 3
    assert points.dtype in (np.float32, np.float64)

    simpleply.write_point_cloud(plyname, selected_points, selected_img)

    assert Path(plyname).exists()
