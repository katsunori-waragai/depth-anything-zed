FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# for depth anything
RUN apt update
RUN apt install sudo
RUN apt install -y zip
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt install -y libv4l-dev v4l-utils qv4l2
RUN apt install -y curl
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install zstd
RUN python3 -m pip install matplotlib numpy 
RUN python3 -m pip install opencv-python 
RUN python3 -m pip install huggingface_hub onnxruntime onnx
# only for development
RUN apt update && apt install -y eog nano

ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent


# for depth anything
RUN cd /root && git clone https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin
RUN cd /root/Depth-Anything-for-Jetson-Orin
WORKDIR /root/Depth-Anything-for-Jetson-Orin
COPY *.py ./
COPY reinstall-opencv.sh ./
RUN cd  /root/Depth-Anything-for-Jetson-Orin
