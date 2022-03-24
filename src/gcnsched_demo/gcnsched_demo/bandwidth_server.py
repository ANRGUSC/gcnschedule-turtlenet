from interfaces.srv import Bandwidth

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(Bandwidth, 'bandwidth_service', self.bandwidth_server_callback)

    def bandwidth_server_callback(self, request, response):
        response.b = "response from service"
        self.get_logger().info('Incoming request\na: %s' % request.a)

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
