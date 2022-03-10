import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from interfaces.msg import Num # custom msg type


class Scheduler(Node):

    def __init__(self):
        super().__init__('scheduler')
        self.publisher_ = self.create_publisher(Num, 'scheduler_status', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Num()
        msg.num = self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing scheduler status: "%d"' % msg.num)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    gcn_sched = Scheduler()

    rclpy.spin(gcn_sched)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gcn_sched.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
