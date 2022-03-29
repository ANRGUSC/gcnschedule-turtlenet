FROM smile_ros
ARG DEBIAN_FRONTEND=noninteractive

COPY rosboard /workspace/rosboard

RUN apt update && apt install -y python3-pip
RUN pip3 install tornado && pip3 install simplejpeg && pip3 install rospkg