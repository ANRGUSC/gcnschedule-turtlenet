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
