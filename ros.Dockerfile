FROM ros:dashing
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y python3-colcon-common-extensions libjpeg-dev zlib1g-dev python3-pip
RUN pip3 install matplotlib networkx torch
RUN pip3 install git+https://github.com/ANRGUSC/edGNN.git@feature/gcnsched
WORKDIR /workspace 