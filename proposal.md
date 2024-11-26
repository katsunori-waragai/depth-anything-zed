# Proposal: Depth estimation from C1 camera.
## Assumption
- C1 camera from Tier IV
- Depth Anything

## Proposal
- Create a camera library that can provide depth estimation with some basis for scaling with a monocular C1 camera.

## Reason for proposal.
- Stereo cameras (including active stereo) that require a baseline length are larger than they should be.
  - In cases such as near the robot's wrist, the camera should be small and light.
- If the camera is so close that only one of the cameras can capture the image, depth cannot be calculated with most stereo cameras.
  - When considering the use of a stereo camera for a robot hand, it is difficult to design a system that cannot detect an object when it is too close.

## How to realize the proposal
#### Make a C1 camera stereo camera
- Select the viewing angle of the camera.
- Define the depth range to be measured.
- Determine the baseline length to cover that range.
- Mount the two C1 cameras stably at the baseline length. (The accuracy and stability of the orientation of the cameras when mounted is also important.)
- Calibrate the stereo cameras with the two C1 cameras.
#### Calculate the parallax with the C1 camera stereo cameras.
- Calculate parallax as a stereo camera with the C1 camera stereo camera.
- Calculate the disparity in depth-anything from the left camera image of the C1 camera.
- Correspondence.
- Calculate disparity from the C1 camera left camera image and create a processing flow as a 3D point cloud as a stereo camera.
#### Create a processing flow for the C1 camera monocular.
- With the above work, a processing flow to obtain 3D point cloud coordinates has been created using only the left camera and depth-anything.
- Execute it for the monocular camera.

## Aiming range
- Front-back relation of the objects must be maintained
- No large missing values
## Do not aim for
- Accuracy of measurements
- Fix target coordinates with a single measurement and no feedback during the measurement
