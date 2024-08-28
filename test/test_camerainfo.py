import inspect
import pyzed.sl as sl

from depanyzed.camerainfo import get_fx_fy_cx_cy, get_baseline


def test_get_baseline():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set up initial parameters for the camera
    init_params = sl.InitParameters()

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    # Retrieve camera information
    cam_info = zed.get_camera_information()
    baseline = get_baseline(cam_info)
    assert 110 < baseline < 130
    zed.close()


def test_get_fx_fy_cx_cy():

    # Create a ZED camera object
    zed = sl.Camera()

    # Set up initial parameters forget_baseline(cam_info)()

    init_params = sl.InitParameters()
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    # Retrieve camera information
    cam_info = zed.get_camera_information()

    # # Access left and right camera parameters
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam

    fx, fy, cx, cy = get_fx_fy_cx_cy(left_cam_params)

    assert isinstance(fx, float), "fx は float 型であるべきです"
    assert isinstance(fy, float), "fy は float 型であるべきです"
    assert isinstance(cx, float), "cx は float 型であるべきです"
    assert isinstance(cy, float), "cy は float 型であるべきです"

    assert fx > 0, "fx は 0 より大きいべきです"
    assert fy > 0, "fy は 0 より大きいべきです"
    assert cx > 0, "cx は 0 より大きいべきです"
    assert cy > 0, "cy は 0 より大きいべきです"

    zed.close()
