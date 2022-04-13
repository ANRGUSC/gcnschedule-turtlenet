FROM ros:dashing
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y python3-colcon-common-extensions libjpeg-dev zlib1g-dev python3-pip
RUN pip3 install matplotlib networkx torch
RUN pip3 install git+https://github.com/ANRGUSC/edGNN.git@feature/gcnsched
RUN pip3 install --upgrade pip && pip3 install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y ros-dashing-cv-bridge
WORKDIR /workspace 