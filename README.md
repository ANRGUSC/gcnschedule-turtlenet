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

[note] You can run `ros2 run gcnsched_demo service --ros-args -r __ns:=/tb1` to run the bandwidth echo service under the namespace `tb1` from the command line. Then the client usage will need to look like `self.create_client(Bandwidth, 'tb1/bandwidth_service', callback_group=cb_group)` to use this service.
