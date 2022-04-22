from functools import partial
from pprint import pformat
import random
from threading import Thread
import time
from typing import Dict, List, Any, Tuple


import rclpy
from rclpy.node import Node, Client, Publisher
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Image
        # std_msgs/Header header
        # uint32 height
        # uint32 width
        # string encoding
        # uint8 is_bigendian
        # uint32 step
        # uint8[] data

from interfaces.srv import Executor
import json
from uuid import uuid4
import os
from itertools import product

from .task_graph import TaskGraph, get_graph

from copy import deepcopy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

BYTES_SENT = 1000

class Visualizer(Node):
    def __init__(self,
                 nodes: List[str],
                 interval: int) -> None:
        super().__init__('visualizer')
        #getting parameters from the launch file
        self.declare_parameter('nodes', [])
        self.declare_parameter('interval', 10)
        nodes = self.get_parameter('nodes').get_parameter_value().string_array_value
        self.interval = self.get_parameter('interval').get_parameter_value().integer_value
        self.all_nodes = nodes

        self.get_logger().info("VISUALIZER INIT")

        self.bandwidths: Dict[Tuple[str, str], float] = {}
        for src, dst in product(nodes, nodes):
            self.create_subscription(
                Float64, f"/{src}/{dst}/delay",
                partial(self.delay_callback, src, dst)
            )

        self.network_publisher: Publisher = self.create_publisher(Image, "/network")
        self.create_timer(self.interval, self.draw_network)

        self.current_tasks: Dict[str, str] = {}
        for node in nodes:
            self.create_subscription(
                String, f"/{node}/current_task",
                partial(self.current_task_callback, node)
            )

    def delay_callback(self, src: str, dst: str, msg: Float64) -> None:
        # converting delays to bandwidths 
        bandwidth =  (BYTES_SENT/1000) / (msg.data/1000) if msg.data != 0.0 else 0.0
        self.bandwidths[(src, dst)] = bandwidth

    def current_task_callback(self, node: str, msg: String) -> None:
        self.current_tasks[node] = msg.data
        
    def get_bandwidth(self, n1: str, n2: str) -> float:
        now = time.time()
        ptime_1, bandwidth_1 = self.bandwidths.get((n1,n2), (0, 0))
        ptime_2, bandwidth_2 = self.bandwidths.get((n2,n1), (0, 0))
        if ptime_1 > ptime_2:
            bandwidth = bandwidth_1 if (now - ptime_1) < 10 else 0
        else:
            bandwidth = bandwidth_2 if (now - ptime_2) < 10 else 0

        return bandwidth + 1e-9

    def draw_network(self) -> None:
        self.get_logger().info("Bandwidths:"+pformat(self.bandwidths))
        self.get_logger().info("Assignment:"+pformat(self.current_tasks))

        graph = nx.Graph()
        edge_labels = {}
        # Code to take the average of the values but assigns 0 if any of the values is 0
        # for src,dst in self.bandwidths: 
        #     avg = (self.bandwidths[(src,dst)] + self.bandwidths[(dst,src)])/2
        #     edge_labels[(src,dst)] = (round(avg, 2)
        #                                 if self.bandwidths[(src,dst)] != 0.0 and self.bandwidths[(dst,src)] != 0.0 
        #                                 else 0.0)
        
        # Code to take the minimum
        for src,dst in self.bandwidths: 
            edge_labels[(src,dst)] = round(min(self.bandwidths[(src,dst)], self.bandwidths[(dst,src)]),2)
        
        self.get_logger().info("EDGE:"+pformat(edge_labels))

        self.get_logger().info("STARTING add weights")
        
        # edge_labels = {
        #     (src, dst): f"{self.get_bandwidth(src, dst):0.2f}"
        #     for src, dst in product(self.all_nodes, self.all_nodes)
        # }
        graph.add_weighted_edges_from(
            [(src, dst, bw) for (src, dst), bw in edge_labels.items()]
        )
       
        pos = nx.planar_layout(graph)
        fig = plt.figure()
        nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=1500, node_color='tab:blue', alpha=0.9,
            with_labels = True, font_weight = 'bold'
        )
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_color='red'
        )

        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # self.get_logger().info(str(img))

        bridge = CvBridge()
        img_msg = Image()
        # img_msg.header.stamp = self.get_clock().now()
        img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")

        self.network_publisher.publish(img_msg)
        self.get_logger().info("publishing network image for visualization")
        plt.close(fig)


def main(args=None):
    rclpy.init(args=args)

    all_nodes = []

    gcn_sched = Visualizer(
        nodes=all_nodes,
        interval=1
    )

    try:
        rclpy.spin(gcn_sched)
    finally:
        gcn_sched.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
