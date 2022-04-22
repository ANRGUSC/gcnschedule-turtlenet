from interfaces.msg import StringStamped
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node, Publisher
# from rclpy.callback_groups import ReentrantCallbackGroup
# from rclpy.executors import MultiThreadedExecutor
import time
from functools import partial
from typing import List
from rclpy.time import Time

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile, QoSPresetProfiles

PING_MESSAGE = "hello"*10000
class BandwidthNode(Node):
    def __init__(self) -> None:
        super().__init__("bandwidth")
        self.PING_MSG = "hello"*100

        self.declare_parameter('name', 'default_node')
        self.declare_parameter('other_nodes',[])
        self.declare_parameter('interval', 5)
        self.interval = self.get_parameter('name').get_parameter_value().double_value
        self.name = self.get_parameter('name').get_parameter_value().string_value
        self.other_nodes = self.get_parameter('other_nodes').get_parameter_value().string_array_value
        self.get_logger().debug(f"INIT {self.name}")
        if not self.other_nodes:
            self.get_logger.info("Error: could not find other nodes")

        my_ns = self.get_namespace().split("/")[-1]
        self.request_topics = {}
        self.response_topics = {}
        self.publish_topics = {}

        # self.qos_profile = QoSProfile(depth=1)
        # self.qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        # self.qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        # self.qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        for other_node in self.other_nodes:
            self.create_subscription(StringStamped, f"/{other_node}/{my_ns}/ping",partial(self.send_reply, other_node, my_ns),qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
            self.create_subscription(StringStamped, f"/{my_ns}/{other_node}/reply",partial(self.publish_metric, other_node, my_ns),qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
            self.request_topics[other_node] = self.create_publisher(StringStamped, f"/{my_ns}/{other_node}/ping", qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
            self.response_topics[other_node] = self.create_publisher(StringStamped, f"/{other_node}/{my_ns}/reply", qos_profile=QoSPresetProfiles.SENSOR_DATA.value)
            self.publish_topics[other_node] = self.create_publisher(Float64, f"/{my_ns}/{other_node}/bandwidth", qos_profile=QoSPresetProfiles.SENSOR_DATA.value)

        self.create_timer(self.interval, self.ping_node)

    def ping_node(self):
        # ping everyone (requests)
        for other_node in self.other_nodes:
            s = StringStamped()
            s.header.stamp = self.get_clock().now().to_msg()
            s.string = self.PING_MSG
            self.request_topics[other_node].publish(s)

    def send_reply(self, other_node, my_ns, msg):
        # reply to a ping request
        s = StringStamped()
        s.header = msg.header # repeat header, so reply has time of initial sent
        s.string = msg.string
        self.response_topics[other_node].publish(s)

    def publish_metric(self, other_node, my_ns, msg):
        # process a ping reply
        self.get_logger().debug(str(self.get_clock().now()))
        self.get_logger().debug(str(msg.header.stamp))
        dt =  self.get_clock().now() - Time.from_msg(msg.header.stamp)
        dt_secs = dt.nanoseconds/1e9
        self.get_logger().debug(f'TIME For Ping {my_ns} to {other_node} roundtrip:{dt_secs}')
        f = Float64()
        req_length = len(self.PING_MSG.encode("utf-8"))
        f.data = (req_length / dt_secs)/125
        self.publish_topics[other_node].publish(f)

def main(args=None):
    rclpy.init(args=args)

    bandwidth_node = BandwidthNode()

    rclpy.spin(bandwidth_node)
    bandwidth_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
