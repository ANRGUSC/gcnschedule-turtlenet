FROM ros:dashing
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3-colcon-common-extensions
WORKDIR /workspace 