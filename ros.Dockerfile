FROM ros:dashing
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y python3-colcon-common-extensions libjpeg-dev zlib1g-dev python3-pip ffmpeg libsm6 libxext6 ros-dashing-cv-bridge
RUN pip3 install matplotlib networkx torch \
    && pip3 install git+https://github.com/ANRGUSC/edGNN.git@feature/gcnsched \
    && pip3 install --upgrade pip && pip3 install opencv-python \
    && pip3 install wfcommons

RUN apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config iputils-ping \
    && pip3 install graphviz pygraphviz
RUN pip3 install git+https://github.com/ANRGUSC/heft.git && pip3 install psutil
WORKDIR /workspace 