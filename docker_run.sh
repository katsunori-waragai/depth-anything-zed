#!/bin/bash
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	-v ~/github/yolox-docker/YOLOX_outputs:/root/YOLOX/YOLOX_outputs \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix depth-anything-zed:100
 
