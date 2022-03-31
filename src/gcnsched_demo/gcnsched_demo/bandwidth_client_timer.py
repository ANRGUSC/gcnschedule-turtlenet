# https://github.com/ros2/examples/blob/dashing/rclpy/services/minimal_client/examples_rclpy_minimal_client/client_async_callback.py

# Copyright 2018 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from interfaces.srv import Bandwidth

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import time


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('minimal_client')
    # Node's default callback group is mutually exclusive. This would prevent the client response
    # from being processed until the timer callback finished, but the timer callback in this
    # example is waiting for the client response
    cb_group = ReentrantCallbackGroup()
    cli = node.create_client(Bandwidth, 'bandwidth_service', callback_group=cb_group)
    did_run = False
    did_get_result = False

    async def call_service():
        nonlocal cli, node, did_run, did_get_result
        did_run = True
        t0 = time.perf_counter()
        try:
            req = Bandwidth.Request()
            req.a = "hello"
            future = cli.call_async(req)
            result = await future
            if result is not None:
                t1 = time.perf_counter()
                node.get_logger().info(
                    'Result of service: for %s = %s; Time0 = %d, Time1 = %d Time taken = %f' %
                    (req.a, result.b, t0, t1, t1 - t0))
            else:
                node.get_logger().warning('Service call failed %r' % (future.exception(),))
        finally:
            did_get_result = True

    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().warning('service not available, waiting again...')

    timer = node.create_timer(5, call_service, callback_group=cb_group)  # every 5 seconds

    # while rclpy.ok() and not did_run:
    #     rclpy.spin_once(node)
    #
    # # if did_run:
    # #     # call timer callback only once
    # #     timer.cancel()
    #
    # while rclpy.ok() and not did_get_result:
    #     rclpy.spin_once(node)

    while rclpy.ok():
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
