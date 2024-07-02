FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "8.7+PTX"
ENV CUDA_HOME /usr/local/cuda-11.4/
RUN cd /root && git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

WORKDIR /root
RUN apt update && apt install -y --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* \
    vim=2:* \
    zstd
RUN apt install -y python3-tk
RUN apt clean -y && apt autoremove -y && rm -rf /var/lib/apt/lists/*
# for depth anything
RUN apt update
RUN apt install sudo
RUN apt-get install -y zip
RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get install -y libv4l-dev v4l-utils qv4l2
RUN apt-get install -y curl
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
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
RUN mkdir weights/
COPY weights/* ./weights/
COPY copyto_host.sh ./
RUN cd  /root/Depth-Anything-for-Jetson-Orin
