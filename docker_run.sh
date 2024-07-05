!/bin/bash
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	-v ${GIT_ROOT}/depth-anything-zed/weights:/root/Depth-Anything-for-Jetson-Orin/weights \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix depth-anything-zed:100
 
