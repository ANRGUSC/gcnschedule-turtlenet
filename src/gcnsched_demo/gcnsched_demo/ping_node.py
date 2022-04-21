from asyncio.subprocess import PIPE
from code import interact
from lib2to3.pytree import Node
import subprocess
import time
import rclpy
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

list_of_ip = ['rpi-tb1', 'rpi-tb2', 'rpi-tb3', 'rpi-tb4']

class PingNode(Node):
    def __init__(self, name: str, other_nodes: List[str], interval: float) ->None:
        super().__init__("ping")
        self.declare_parameter('name', 'default_node')
        self.declare_parameter('other_nodes',[])
        self.declare_parameter('interval',5)
        self.interval = self.get_parameter('interval').get_parameter_value().double_value
        name = self.get_parameter('name').get_parameter_value().string_value
        other_nodes = self.get_parameter('other_nodes').get_parameter_value().string_array_value
        self.get_logger().info(f"INIT {name}")
        self.numPings = 1
        self.pubDict = {}
        if not other_nodes:
            self.get_logger.info("Error: could not find other nodes")
        
        cb_group = ReentrantCallbackGroup()
        self.name = name
        self.IPList = [ip for ip in list_of_ip if name[-1] not in ip]
        print(self.IPList)
        for other_node in other_nodes:
            self.pubDict[other_node] = self.create_publisher(Float64, f'/{other_node}/delay',10, callback_group=cb_group)
        self.create_timer(self.interval, self.timer_callback)
        # self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        avgValues = {}
        start = time.time()
        # self.get_logger().info("In timer callback")
        for ip in self.IPList:
            result = subprocess.run(
            # Command as a list, to avoid shell=True
            ['ping', '-c', str(self.numPings),'-s','1000', ip],
            stdout=PIPE
            )
            avgValues[ip] = 0
            for line in result.stdout.splitlines():
                line = line.decode("utf-8")
                if "icmp_seq" in line:                    
                    timing = line.split('time=')[-1].split(' ms')[0]
                    try:
                        avgValues[ip] += float(timing)
                    except ValueError:
                        self.get_logger().info('Failed Ping')
                        # print("Failed Ping")
                    except Exception as ex:
                        self.get_logger().info(f'Exception: {type(ex).__name__}')
        time_taken = time.time() - start
        print("Time taken ", time_taken)
        for ip in avgValues:
            self.get_logger().info(f'{ip} {avgValues[ip]/self.numPings}')
            print(ip, avgValues[ip]/self.numPings)
        # TODO: Add publishing code here


def main(args=None):
    rclpy.init(args=args)

    name = "default-node"
    all_nodes = []

    ping_node = PingNode(
        name=name,
        other_nodes=[node for node in all_nodes if node != name],
        interval=5
    )
    rclpy.spin(ping_node)
    ping_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()