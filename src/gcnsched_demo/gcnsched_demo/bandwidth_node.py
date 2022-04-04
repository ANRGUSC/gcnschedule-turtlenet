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
        name = self.get_parameter('name').get_parameter_value().string_value
        other_nodes = self.get_parameter('other_nodes').get_parameter_value().string_array_value
        # m_param = self.get_parameter('my_parameter').get_parameter_value().string_value
        # print(m_param)
        # self.get_logger().info('Hello %s!' % name)
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
            pub = self.create_publisher(Float64, f"/{other_node}/bandwidth", 10, callback_group=cb_group)
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(f'service {other_node}/ping not available, waiting again...')
            self.create_timer(interval, partial(self.ping_node, cli, pub), callback_group=cb_group)

    def ping_node(self, cli: Client, pub: Publisher) -> None:
        MSG = "hello"
        
        req = Bandwidth.Request()
        req.a = MSG
        start = time.time()
        res: Bandwidth.Response = cli.call(req)
        dt = time.time() - start
        msg = Float64()
        msg.data = len(MSG.encode("utf-8")) / dt
        pub.publish(msg)

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
    
    executor = MultiThreadedExecutor()
    executor.add_node(bandwidth_client_node)

    executor.spin()
    bandwidth_client_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
