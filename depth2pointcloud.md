# 点群への変換
- 数値としてのdepthを点群データに変換して妥当性を確認しやすくすること
- 以下のissue を読むと点群データへの変換と可視化の例が記されている。
https://github.com/LiheYoung/Depth-Anything/issues/36

## linearの軸で算出した結果を点群に変換すること
```commandline
@dataclass
class Depth2Points:
    fx: float
    fy: float
    cx: float
    cy: float
    def cloud_points(self, depth):
        pass
        
```

## ZED SDK での撮影の解像度のモードを確認すること
それによって、焦点距離の情報が変わってくることを考慮すること。

depthからpointcloud に変化する手続きの例は、以下のURLを参照した。
https://github.com/LiheYoung/Depth-Anything/issues/36


以下のような手続きで変更可能なはずである。
ここで、以下の値が既知であることを前提としている。

focal_length_x: float, focal_length_y: float



```

def to_point_cloud_np(resized_pred: np.ndarray, focal_length_x: float, focal_length_y: float) -> np.ndarray:
    """
    """
    height, width = resized_pred.shape[:2]
    P_x = width // 2 # center of the image
    P_y = height // 2
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - P_x) / focal_length_x
    y = (y - P_y) / focal_length_y
    z = np.array(resized_pred)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    return points
```

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
