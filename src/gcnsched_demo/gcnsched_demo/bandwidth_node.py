import asyncio
from interfaces.srv import Bandwidth
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node, Client, Publisher
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
from functools import partial
from typing import List
import os
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType


class BandwidthNode(Node):
    def __init__(self, name: str, other_nodes: List[str], interval: float) -> None:
        super().__init__("bandwidth") #f"{name}_bandwidth")
        self.declare_parameter('name', 'default_node')
        self.declare_parameter('other_nodes',[])
        self.interval = interval
        print("INTERVAL", interval)
        name = self.get_parameter('name').get_parameter_value().string_value
        other_nodes = self.get_parameter('other_nodes').get_parameter_value().string_array_value
        self.get_logger().info(f"INIT {name}")
        if not other_nodes:
            self.get_logger.info("Error: could not find other nodes")

        cb_group = ReentrantCallbackGroup()
        self.ping_service = self.create_service(
            Bandwidth, 'ping',
            self.ping_callback,
            callback_group=cb_group
        )

        self.name = name
        for other_node in other_nodes:
            cli = self.create_client(Bandwidth, f"/{other_node}/ping", callback_group=cb_group)
            pub = self.create_publisher(Float64, f"{other_node}/bandwidth", 10, callback_group=cb_group)
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(f'service {other_node}/ping not available, waiting again...')
            self.create_timer(interval, partial(self.ping_node, cli, pub), callback_group=cb_group)

    def publish_ping(self, start: float, pub: Publisher, req_length: int, *args, **kwargs) -> None:
        self.get_logger().debug("UNSTUCK")
        dt = time.time() - start
        msg = Float64()
        msg.data = req_length / dt # take message size into account
        self.get_logger().debug("publishing")
        pub.publish(msg)

    def ping_node(self, cli: Client, pub: Publisher) -> None:
        self.get_logger().debug("Inside PING NODE")
        MSG = "hello"*1000
        req_length = len(MSG.encode("utf-8"))
        req = Bandwidth.Request()
        req.a = MSG
        start = time.time()
        self.get_logger().debug("STUCK")
        fut = cli.call_async(req)
        fut.add_done_callback(partial(self.publish_ping, start, pub, req_length))

    def ping_callback(self,
                      request: Bandwidth.Request,
                      response: Bandwidth.Response) -> Bandwidth.Response:
        response.b = request.a
        return response

def main(args=None):
    rclpy.init(args=args)

    name = "default-node"
    all_nodes = []

    bandwidth_client_node = BandwidthNode(
        name=name,
        other_nodes=[node for node in all_nodes if node != name],
        interval=5
    )

    # executor = MultiThreadedExecutor(num_threads=1000)
    # executor.add_node(bandwidth_client_node)

    # executor.spin()
    rclpy.spin(bandwidth_client_node)
    bandwidth_client_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
