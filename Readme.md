# ROS GCN

## Setup
If you opened with Gitpod, you can skip the Setup section!

First build the docker images:
```bash
docker build -t smile_ros -f ros.Dockerfile .
docker build -t rosboard -f rosboard.Dockerfile .
```

Then pull and run Dozzle, a Docker image that's helpful for viewing logs:
```bash
docker pull amir20/dozzle:latest
docker run --name dozzle -d --volume=/var/run/docker.sock:/var/run/docker.sock -p 8888:8080 amir20/dozzle:latest
```

## Running
```bash
docker-compose up -d
```

# gcnschedule-turtlenet
Graph Convolutional Network-based Scheduler for Distributing Computation in the Internet of Robotic Things

https://smile-sdsu.github.io/cps_iot22/


From the `ws` run:

`colcon build --packages-select gcnsched_demo`

Then for example:

`ros2 run gcnsched_demo sample_node`

If you have any trouble try (from the `ws`):
`. install/setup.bash`

To make a new ROS2 node, make a new file under `gcnsch_demo/gcnsched_demo` and then add a line to the entry points in `setup.py`.
