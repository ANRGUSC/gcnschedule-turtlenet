import sys

from interfaces.srv import Bandwidth
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import time


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        cb_group = ReentrantCallbackGroup()
        self.cli = self.create_client(Bandwidth, 'bandwidth_service', callback_group=cb_group)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Bandwidth.Request()
        timer = self.create_timer(1, self.call_service, callback_group=cb_group)  # every 5 seconds

    # def send_request(self):
    #     self.req.a = "hello"
    #     global t0
    #     t0 = time.perf_counter()
    #     self.future = self.cli.call_async(self.req)

    async def call_service(self):
        t0 = time.perf_counter()
        try:
            self.req.a = "hello"
            future = self.cli.call_async(self.req)
            result = await future
            if result is not None:
                t1 = time.perf_counter()
                self.get_logger().info(
                    'Result of service: for %s = %s; Time0 = %d, Time1 = %d Time taken = %f' %
                    (self.req.a, result.b, t0, t1, t1 - t0))
            else:
                self.get_logger().info('Service call failed %r' % (future.exception(),))
        finally:
            pass

def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        # if minimal_client.future.done():
        #     try:
        #         response = minimal_client.future.result()
        #     except Exception as e:
        #         minimal_client.get_logger().info(
        #             'Service call failed %r' % (e,))
        #     else:
        #         t1 = time.perf_counter()
        #         minimal_client.get_logger().info(
        #             'Result of service: for %s = %s; Time0 = %d, Time1 = %d Time taken = %f' %
        #             (minimal_client.req.a, response.b, t0, t1, t1 - t0))
        #     break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
