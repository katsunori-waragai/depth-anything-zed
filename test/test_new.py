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
    pass
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
    # right_cam_params = cam_info.camera_configuration.calibration_parameters.right_cam



    # Print some of the camera parameters
    # print("Left Camera Parameters:")
    print(f"image_size: {left_cam_params.image_size.width} x {left_cam_params.image_size.height}")
    print(f"Focal Length (fx, fy): {left_cam_params.fx}, {left_cam_params.fy}")
    print(f"Principal Point (cx, cy): {left_cam_params.cx}, {left_cam_params.cy}")
    print(f"Distortion Coefficients: {left_cam_params.disto}")

    # print("\nRight Camera Parameters:")
    # print(f"image_size: {right_cam_params.image_size.width} x {right_cam_params.image_size.height}")
    # print(f"Focal Length (fx, fy): {right_cam_params.fx}, {right_cam_params.fy}")
    # print(f"Principal Point (cx, cy): {right_cam_params.cx}, {right_cam_params.cy}")
    # print(f"Distortion Coefficients: {right_cam_params.disto}")
    # print("\n")
    # print(f"{cam_info.camera_configuration.calibration_parameters.get_camera_baseline()=}")

    fx, fy, cx, cy = get_fx_fy_cx_cy(left_cam_params)
    
    assert isinstance(fx,float)
    assert 0< cx < left_cam_params.image_size.width
    # print(f"{get_baseline(cam_info)}")
    # # Close the camera
    zed.close()
