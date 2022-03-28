FROM gitpod/workspace-full

SHELL ["bash", "-c"]
USER root
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y ros-foxy-desktop
RUN DEBIAN_FRONTEND=noninteractive apt install -y libpython3-dev python3-pip \
    && pip3 install -U argcomplete

USER gitpod
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
