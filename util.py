import cv2
import numpy as np


def finitemax(depth: np.ndarray):
    return np.nanmax(depth[np.isfinite(depth)])


def finitemin(depth: np.ndarray):
    return np.nanmin(depth[np.isfinite(depth)])


def depth_as_colorimage(depth_raw: np.ndarray, vmin=None, vmax=None, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    apply color mapping with vmin, vmax
    """
    if vmin is None:
        vmin = finitemin(depth_raw)
    if vmax is None:
        vmax = finitemax(depth_raw)
    depth_raw = (depth_raw - vmin) / (vmax - vmin) * 255.0
    depth_raw = depth_raw.astype(np.uint8)  # depth_raw might have NaN, PosInf, NegInf.
    return cv2.applyColorMap(depth_raw, colormap)


def depth_as_gray(depth_raw: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    apply color mapping with vmin, vmax
    """
    if vmin is None:
        vmin = finitemin(depth_raw)
    if vmax is None:
        vmax = finitemax(depth_raw)
    depth_raw = (depth_raw - vmin) / (vmax - vmin) * 255.0
    gray = depth_raw.astype(np.uint8)  # depth_raw might have NaN, PosInf, NegInf.
    return cv2.merge((gray, gray, gray))


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))


def as_matrix(chw_array: np.ndarray) -> np.ndarray:
    H_, W_ = chw_array.shape[-2:]
    return np.reshape(chw_array, (H_, W_))
