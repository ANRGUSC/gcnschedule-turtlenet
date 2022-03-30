import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from interfaces.msg import Num


class SampleNode(Node):

    def __init__(self):
        super().__init__('sample_node')
        self.publisher_ = self.create_publisher(String, 'node_status', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        self.subscriber_ = self.create_subscription(Num, 'scheduler_status', self.scheduler_status_callback, 10)
        self.subscriber_

    def timer_callback(self):
        msg = String()
        msg.data = 'Sample node status %d: GOOD' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def scheduler_status_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.num)


def main(args=None):
    rclpy.init(args=args)

    sample_node = SampleNode()

    rclpy.spin(sample_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sample_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
