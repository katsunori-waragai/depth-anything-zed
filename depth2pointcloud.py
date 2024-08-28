"""
# Open3D を使う場合
このファイルのモジュールは不要になる。
https://www.open3d.org/
https://pypi.org/project/open3d/
https://github.com/isl-org/Open3D
https://www.open3d.org/docs/release/

# 点群への変換
- 数値としてのdepthを点群データに変換して妥当性を確認しやすくすること
- 以下のissue を読むと点群データへの変換と可視化の例が記されている。
https://github.com/LiheYoung/Depth-Anything/issues/36

## linearの軸で算出した結果を点群に変換すること

## ZED SDK での撮影の解像度のモードを確認すること
それによって、焦点距離の情報が変わってくることを考慮すること。

depthからpointcloud に変化する手続きの例は、以下のURLを参照した。
https://github.com/LiheYoung/Depth-Anything/issues/36


以下のような手続きで変更可能なはずである。
ここで、以下の値が既知であることを前提としている。

https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view

Focal Length in pixels
Note: The exact focal length can vary depending on the selected resolution, and the camera calibration. We recommend using runtime information from the ZED SDK.

https://github.com/stereolabs/zed-opencv/issues/39


/usr/local/zed/settings/SN[0-9]+.conf

```commandline

head -22 SN*.conf
[LEFT_CAM_2K]
fx=1064.82
fy=1065.07
cx=1099.05
cy=628.813
k1=-0.0634518
k2=0.0401532
p1=-0.000375119
p2=0.00074809
k3=-0.0161231

[RIGHT_CAM_2K]
fx=1065.27
fy=1065.34
cx=1133.31
cy=654.957
k1=-0.0587747
k2=0.0322036
p1=5.85653e-05
p2=-0.000297978
k3=-0.0123602
```
"""

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Depth2Points:
    fx: float
    fy: float
    cx: float
    cy: float

    def cloud_points(self, depth):
        """
        ここでは、depthが2次元配列であることが必要
        """
        H_, W_ = depth.shape[:2]
        x, y = np.meshgrid(np.arange(W_), np.arange(H_))
        assert x.shape == depth.shape
        x = (x - self.cx) / self.fx
        y = (y - self.cy) / self.fy
        z = np.array(depth)  # [mm]
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        return points


def disparity_to_depth(disparity: np.ndarray, baseline=119.987, focal_length=532.41) -> np.ndarray:
    """
    disparity(視差)をdepth(深度）に変換する。



        fx = 532.41
        fy = 532.535
        cx = 636.025  # [pixel]
        cy = 362.4065  # [pixel]
    """
    return baseline * focal_length / disparity


def depth_to_disparity(depth: np.ndarray, baseline=119.987, focal_length=532.41) -> np.ndarray:
    """
    depth(深度）をdisparity(視差)に変換する。



        fx = 532.41
        fy = 532.535
        cx = 636.025  # [pixel]
        cy = 362.4065  # [pixel]
    """
    return baseline * focal_length / depth


if __name__ == "__main__":
    import argparse
    import simpleply
    parser = argparse.ArgumentParser("depth npy to point cloud")
    parser.add_argument("--depth", default="data/zed_depth.npy", help="depth npy file")
    parser.add_argument("--rgb", default="data/left.png", help="left rgb image file")
    args = parser.parse_args()

    # depth_file = "data/zed_depth.npy"
    # rgb_file = "data/left.png"
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

    # 点群の座標の原点を移動して、meshlab での表示を楽にする。
    mean_point = np.mean(selected_points, axis=0)

    centered_points = selected_points.copy()
    centered_points[:, 0] -= mean_point[0]
    centered_points[:, 1] -= mean_point[1]
    centered_points[:, 2] -= mean_point[2]
    plyname2 = "data/test_centered.ply"

    simpleply.write_point_cloud(plyname, selected_points, selected_img)
    simpleply.write_point_cloud(plyname2, centered_points, selected_img)
