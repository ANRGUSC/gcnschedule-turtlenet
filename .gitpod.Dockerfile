FROM gitpod/workspace-full
ARG DEBIAN_FRONTEND=noninteractive

USER gitpod

RUN sudo apt update && sudo apt install curl gnupg2 lsb-release \
    && sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN mkdir -p ~/ros2_foxy \
    && cd ~/ros2_foxy \
    && tar xf ~/Downloads/ros2-package-linux-x86_64.tar.bz2

RUN sudo apt update \
    && sudo apt install -y python3-rosdep \
    && sudo rosdep init \
    && rosdep update

RUN rosdep install --from-paths ~/ros2_foxy/ros2-linux/share --ignore-src -y --skip-keys "cyclonedds fastcdr fastrtps rti-connext-dds-5.3.1 urdfdom_headers"

RUN sudo apt install -y libpython3-dev python3-pip \
    && pip3 install -U argcomplete

RUN echo "source ~/ros2_foxy/ros2-linux/setup.bash" >> ~/.bashrc
