from functools import partial
from pprint import pformat
import random
from threading import Thread
import time
from typing import Dict, List, Any


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
        print(nodes)

        self.get_logger().info("VISUALIZER INIT")

        self.bandwidths: Dict[str, Dict[str, float]] = {}
        for src, dst in product(nodes, nodes):
            self.create_subscription(
                Float64, f"/{src}/{dst}/bandwidth",
                partial(self.bandwidth_callback, src, dst)
            )

        self.network_publisher: Publisher = self.create_publisher(Image, "/network")
        self.create_timer(self.interval, self.draw_network)

        self.current_tasks: Dict[str, str] = {}
        for node in nodes:
            self.create_subscription(
                String, f"/{node}/current_task",
                partial(self.current_task_callback, node)
            )

    def bandwidth_callback(self, src: str, dst: str, msg: Float64) -> None:
        self.bandwidths[(src, dst)] = msg.data
        # print("BANDWIDTHS:", pformat(self.bandwidths))
        # self.get_logger().info("bandwidth callback")

    def current_task_callback(self, node: str, msg: String) -> None:
        self.current_tasks[node] = msg.data

    def draw_network(self) -> None:
        self.get_logger().info("Bandwidths:"+pformat(self.bandwidths))
        self.get_logger().info("Assignment:"+pformat(self.current_tasks))

        graph = nx.Graph()
        # FOR DEBUGGING
        # bandwidths = {("A","B"):10,("A","C"):20}
        bandwidths = deepcopy(self.bandwidths)
        # FOR DEBUGGING 
    #     bandwidths = {('node1', 'node2'): 6.060715075603744e-10,
    #    ('node1', 'node3'): 281.7616552465404,
    #    ('node1', 'node4'): 246.73827872227778,
    #    ('node2', 'node1'): 115.61244797265635,
    #    ('node2', 'node3'): 176.00939991607217,
    #    ('node2', 'node4'): 187.95895137799687,
    #    ('node3', 'node1'): 429.260464640262,
    #    ('node3', 'node2'): 6.060715075446431e-10,
    #    ('node3', 'node4'): 271.56387180317256,
    #    ('node4', 'node1'): 456.10091344062636,
    #    ('node4', 'node2'): 6.060715076076258e-10,
    #    ('node4', 'node3'): 95.15856342310049}
        # print(bwidth)
        self.get_logger().info("STARTING add weights")
        
        bandwidth_set = set()
        updated_bandwidth = {}
        for src,dst in bandwidths.keys():
            if (src,dst) in bandwidth_set or (dst,src) in bandwidth_set:
                continue
            else:
                bandwidth_set.add((src,dst))
        #updating the bandwidths 
        for src,dst in bandwidth_set:
            updated_bandwidth[(src,dst)] = round(min(bandwidths[(src,dst)], bandwidths[(dst,src)]),2)

        self.get_logger().info(pformat(updated_bandwidth))
        graph.add_weighted_edges_from(
            [(src, dst, bw) for (src, dst), bw in updated_bandwidth.items()]
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
            edge_labels=updated_bandwidth,
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
        interval=2
    )

    try:
        rclpy.spin(gcn_sched)
    finally:
        gcn_sched.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
