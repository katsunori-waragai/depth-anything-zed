# depth-anything-zed
docker environment for depth-anything　with ZED SDK

This repository provide scripts using ZED SDK (StereoLabs)
This scripts assume cameras by StereoLabs.

## Aim
### Why depth-anything as-is is not sufficient.
- The result of depth-anything cannot be converted to point cloud data with scale.
- Focal length information such as fx, fy, etc., and baseline values that are assumed to be the calculated parallax equivalent.
- Due to the lack of these values, depth-anything cannot be turned into 3D data in real space.
### What this library aims to do
- To provide a method to convert the 3D coordinates of a real space by depth-anything by using a camera with known characteristics.
- To convert the disparity returned by depth-anything to that of a real stereo camera by robustly fitting the disparity calculated by a camera with known characteristics and the disparity returned by depth-anything.

### なぜ、depth-anything そのままでは不十分なのか
- depth-anythingの結果をスケールの持つ点群データに変換できないこと。
- fx, fy などの焦点距離情報、視差相当の算出値が想定しているbaselineの値
- これらの値がないため、depth-anyting では、実空間の3Dデータにすることができない。
### このライブラリが目指すこと
- 特性のわかっているカメラを使うことで、depth-anything による実空間の3D座標に変換できる方法を提供すること。
- 特性のわかっているカメラで算出した視差と、depth-anythingが返す視差とをロバストにフィッティングすることで、depth-anything が返す視差を、実在のステレオカメラの視差相当に変換すること。


## Checked Environment
- NVIDIA Jetson AGX orin
- Ubuntu 20
- python3.8
- ZED SDK (StereoLabs)
- ZED2i (StereoLabs)

## Install with Docker
```commandline
sh docker_build.sh
sh docker_run.sh
```
# model conversion from onnx model to TRT model
for the first time execute as follows 
```commandline
make model
```

Be patient! 
This command takes 10 minutes or more.
And too many ignorable warnings.
After conversion
```commandline
 ls weights/
depth_anything_vits14_308.onnx  depth_anything_vits14_364.trt   depth_anything_vits14_518.onnx
depth_anything_vits14_308.trt   depth_anything_vits14_406.onnx  depth_anything_vits14_518.trt
depth_anything_vits14_364.onnx  depth_anything_vits14_406.trt
```

## scripts
- usb_depth_anything.py:  depth-anything using ZED2i as USB camera.
- zed_depth_anything.py:   depth-anything using ZED2i with zed sdk.
- zed_scaled_depth_anything.py: scaled depth-anything with help of disparity by zed sdk. 

## depth-anything with USB camera interface
The left and right images are retrieved from the cv2.VideoCapture() interface as one concatenated image.
ZED2i is used as a USB camera.
What you get is the disparity calculated from the left image only.
The disparity values are relative.

```commandline
 python3 usb_depth_anything.py -h
usage: usb_depth_anything.py [-h] [--frame_rate FRAME_RATE] [--raw] [--stream] [--record] [--save] [--grayscale]

depth-anything using zed2i as usb camera

optional arguments:
  -h, --help            show this help message and exit
  --frame_rate FRAME_RATE
                        Frame rate of the camera
  --raw                 Use only the raw depth map
  --stream              Stream the results
  --record              Record the results
  --save                Save the results
  --grayscale           Convert the depth map to grayscale


Script to calculate disparity with depth-anything from images acquired with a ZED2i camera using zed sdk
python3 zed_depth_anything.py -h
usage: zed_depth_anything.py [-h] [--input_svo_file INPUT_SVO_FILE] [--ip_address IP_ADDRESS] [--resolution RESOLUTION]
                             [--confidence_threshold CONFIDENCE_THRESHOLD]

depth-anything(native) with zed camera

optional arguments:
  -h, --help            show this help message and exit
  --input_svo_file INPUT_SVO_FILE
                        Path to an .svo file, if you want to replay it
  --ip_address IP_ADDRESS
                        IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup
  --resolution RESOLUTION
                        Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
  --confidence_threshold CONFIDENCE_THRESHOLD
                        depth confidence_threshold(0 ~ 100)
```

![neural_plus.png](figures/neural_plus.png)
![neural_plus_near.png](figures/neural_plus_near.png)
Examples for  zed_depth_anything.py
Using DEPTH_MODE.NEURAL_PLUS.



```
Script to acquire images from a ZED2i camera using zed sdk and calculate disparity with depth-anything with aligned scaling.
Robust fitting is performed so that the disparity converted from the depth of the stereo calculation in the zed sdk and the disparity in depth-anything are on the same straight line.
 
$ python3 zed_scaled_depth_anything.py -h
usage: zed_scaled_depth_anything.py [-h] [--quick] [--save_depth] [--save_ply] [--save_fullply]

scaled depth-anything

optional arguments:
  -h, --help      show this help message and exit
  --quick         simple output without matplotlib
  --save_depth    save depth and left image
  --save_ply      save ply
  --save_fullply  save full ply

```

- You can compare ZED SDK depth with scaled depth-anything as follows
```commandline
python3 zed_scaled_depth_anything.py --quick
```

## Note
- Even if the objects are extremely close, Depth-Anything can return a depth.
![](figures/depth_anything_example.png)

## helper tools
- use disparity-view to capture and view npy files.
  - https://github.com/katsunori-waragai/disparity-view
  - zed_capture: capture tool 
  - disparity_viewer: disparity npy file viewer

## SEE ALSO
### depth-anything v2
https://github.com/DepthAnything/Depth-Anything-V2
[pdf](https://arxiv.org/abs/2406.09414)

# Acknowlegements
This project is based on following works.
[Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

[Depth-Anything-for-Jetson-Orin](https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin)
