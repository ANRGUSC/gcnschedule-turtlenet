import traceback
from interfaces.srv import Bandwidth
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node, Client, Publisher
from rclpy.callback_groups import ReentrantCallbackGroup
import time
from functools import partial
from typing import List


from .client_timeout import ClientTimeouter

PING_MESSAGE = "hello"*10
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
            pub = self.create_publisher(Float64, f"{other_node}/bandwidth", 10, callback_group=cb_group)

            cli = self.create_client(Bandwidth, f"/{other_node}/ping", callback_group=cb_group)
            cli_timeouter = ClientTimeouter(
                cli,
                timeout=4,
                success_callback=partial(self.publish_ping, pub=pub),
                error_callback=lambda err: self.get_logger().error(str(err)),
            )

            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'service {other_node}/ping not available, waiting again...')

            self.create_timer(interval, partial(self.ping_node, cli_timeouter, pub), callback_group=cb_group)

        self.get_logger().info("Bandwidth node has started!")

    def publish_ping(self, dt: float, res, pub: Publisher, *args, **kwargs) -> None:
        try:
            self.get_logger().info("ping service success callback")
            msg = Float64()
            msg.data = (len(PING_MESSAGE.encode("utf-8")) / dt) / 125000
            self.get_logger().info(f"publishing {dt} to {pub.topic}")
            pub.publish(msg)
        except:
            self.get_logger().error(traceback.format_exc())

    def ping_node(self, cli: ClientTimeouter, pub: Publisher) -> None:
        self.get_logger().debug("Inside PING NODE")
        req = Bandwidth.Request()
        req.a = PING_MESSAGE
        self.get_logger().info("calling ping service")
        cli.call(req)

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
