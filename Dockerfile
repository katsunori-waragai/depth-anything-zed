FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# for depth anything
RUN apt-get update
RUN apt install sudo
RUN apt install -y zip
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt install -y libv4l-dev v4l-utils qv4l2
RUN apt install -y curl
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install zstd
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab

ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent

RUN cd /root && mkdir depth-anything-zed
WORKDIR /root/depth-anything-zed
COPY *.py ./
RUN mkdir ./depanyzed/
COPY ./depanyzed/* ./depanyzed/
COPY pyproject.toml ./
RUN python3 -m pip install .[dev]
COPY Makefile ./
ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
