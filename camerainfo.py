import pyzed.sl as sl

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

# Access left and right camera parameters
left_cam_params = cam_info.calibration_parameters.left_cam
right_cam_params = cam_info.calibration_parameters.right_cam

# Print some of the camera parameters
print("Left Camera Parameters:")
print(f"Resolution: {left_cam_params.resolution.width} x {left_cam_params.resolution.height}")
print(f"Focal Length (fx, fy): {left_cam_params.fx}, {left_cam_params.fy}")
print(f"Principal Point (cx, cy): {left_cam_params.cx}, {left_cam_params.cy}")
print(f"Distortion Coefficients: {left_cam_params.disto}")

print("\nRight Camera Parameters:")
print(f"Resolution: {right_cam_params.resolution.width} x {right_cam_params.resolution.height}")
print(f"Focal Length (fx, fy): {right_cam_params.fx}, {right_cam_params.fy}")
print(f"Principal Point (cx, cy): {right_cam_params.cx}, {right_cam_params.cy}")
print(f"Distortion Coefficients: {right_cam_params.disto}")

# Close the camera
zed.close()
