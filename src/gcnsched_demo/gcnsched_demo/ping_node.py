from asyncio.subprocess import PIPE
from code import interact
from lib2to3.pytree import Node
from pprint import pformat
import random
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
import psutil

list_of_ip = ['rpi-tb1', 'rpi-tb2', 'rpi-tb3', 'rpi-tb4']
adhoc_ip = ['192.168.7.1','192.168.7.2','192.168.7.3','192.168.7.4']

ADHOC = True

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
            self.get_logger().info("Error: could not find other nodes")
        my_ns = self.get_namespace()
        cb_group = ReentrantCallbackGroup()
        self.name = name
        self.IPList = [ip for ip in list_of_ip if name[-1] not in ip]
        if ADHOC:
            self.IPList = [ip for ip in adhoc_ip if name[-1] != ip[-1] ]

        self.get_logger().info(f'{self.IPList}')
        for index, other_node in enumerate(other_nodes):
            self.pubDict[self.IPList[index]] = self.create_publisher(Float64, f'{my_ns}/{other_node}/delay',10, callback_group=cb_group)
        # self.create_timer(self.interval, self.timer_callback)
        self.create_timer(self.interval, self.timer_callback)
        for ip in self.pubDict:
            self.get_logger().info(f"{ip}")
    

    def kill(self, proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def timer_callback(self):
       
        avgValues = {}
        start = time.time()
        self.get_logger().info("In timer callback")
        for ip in self.IPList:
            result = subprocess.Popen(
            # Command as a list, to avoid shell=True
            ['ping', '-c', str(self.numPings),'-s','1000', ip],
            stdout=PIPE
            )
            try:
                result.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.kill(result.pid)
                avgValues[ip] = 0
                continue
            
            # For debugging use random else 0
            # flip = random.randint(0,2)
            avgValues[ip] = 0#round(random.uniform(1,100), 2) if flip < 2 else 0
            # self.get_logger().info(f"result.stdout:" + pformat(result.stdout))
            for line in result.stdout:
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
        self.get_logger().info(f"Time taken {time_taken}")
        for ip in avgValues:
            self.get_logger().info(f'{ip} {avgValues[ip]/self.numPings}')
            print(ip, avgValues[ip]/self.numPings)
            msg = Float64()
            msg.data = avgValues[ip]/self.numPings
         # TODO: Add publishing code here
            self.pubDict[ip].publish(msg)
       
        

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