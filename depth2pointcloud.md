depthからpointcloud に変化する手続きの例は、以下のURLを参照した。
https://github.com/LiheYoung/Depth-Anything/issues/36


以下のような手続きで変更可能なはずである。
ここで、以下の値が基地であることを前提としている。

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

