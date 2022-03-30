FROM smile_ros
ARG DEBIAN_FRONTEND=noninteractive

COPY rosboard /rosboard

RUN apt update && apt install -y python3-pip
RUN pip3 install tornado simplejpeg rospkg psutil