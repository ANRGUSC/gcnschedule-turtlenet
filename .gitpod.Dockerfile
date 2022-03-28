# FROM gitpod/workspace-full
FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

RUN apt update && apt install -y curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update \
    && apt install -y ros-dashing-desktop

RUN apt install -y python3-pip \
    && pip3 install -U argcomplete

USER gitpod
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/dashing/setup.bash 
