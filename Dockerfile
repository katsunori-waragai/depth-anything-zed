FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
RUN apt update
RUN apt install sudo
RUN apt-get install -y zip
RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get install -y libv4l-dev v4l-utils qv4l2
RUN apt-get install -y curl
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt update && apt install -y eog nano

RUN python3 -m pip install -U pip
RUN python3 -m pip install loguru tqdm thop ninja tabulate
RUN python3 -m pip install pycocotools
RUN python3 -m pip install -U jetson-stats 
RUN python3 -m pip install huggingface_hub onnx
RUN cd /root && git clone https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin
RUN cd /root/Depth-Anything-for-Jetson-Orin
WORKDIR /root/Depth-Anything-for-Jetson-Orin
COPY depth_anything*.pth ./
COPY *.py ./
RUN mkdir weights/
COPY weights/ ./
COPY weights/* ./weights/
COPY host_location.txt ./
COPY copyto_host.sh ./
RUN cd  /root/Depth-Anything-for-Jetson-Orin
