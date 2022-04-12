import enum
from functools import partial
from pprint import pformat
import random
from re import I
from threading import Thread
import time
from typing import Dict, List, Any

import torch


import rclpy
from rclpy.node import Node, Client, Publisher
from std_msgs.msg import Float64
from sensor_msgs.msg import Image

from interfaces.srv import Executor
import json
from uuid import uuid4
import os
from itertools import product

from gcnsched.ready_to_use import find_schedule
from .task_graph import TaskGraph, get_graph

# from copy import deepcopy
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from cv_bridge import CvBridge

class Scheduler(Node):
    def __init__(self, 
                 nodes: List[str], 
                 graph: TaskGraph,
                 interval: int) -> None:
        super().__init__('scheduler')
        self.get_logger().info("SCHEDULER INIT")
        self.graph = graph
        self.interval = interval
        self.all_nodes = nodes

        self.executor_clients: Dict[str, Client] = {}
        for node in nodes:
            cli = self.create_client(
                Executor, f'{node}/executor'
            )
            self.executor_clients[node] = cli
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(f'service {node}/executor not available, waiting again...')

        self.bandwidths: Dict[str, Dict[str, float]] = {}
        for src, dst in product(nodes, nodes):
            self.create_subscription(
                Float64, f"{src}/{dst}/bandwidth", 
                partial(self.bandwidth_callback, src, dst)
            )
        self.network_publisher: Publisher = self.create_publisher(Image, "network")
        # self.create_timer(10, self.draw_network)

    def bandwidth_callback(self, src: str, dst: str, msg: Float64) -> None:
        self.bandwidths[(src, dst)] = msg.data

        print("BANDWIDTHS:", pformat(self.bandwidths))

    # def draw_network(self) -> None:
    #     graph = nx.Graph()
    #     bandwidths = deepcopy(self.bandwidths)
    #     graph.add_weighted_edges_from(
    #         [(src, dst, bw) for (src, dst), bw in bandwidths.items()]
    #     )

    #     fig, ax = plt.subplots()
    #     nx.draw_planar(graph, ax=ax)
    #     fig.canvas.draw()

    #     # convert canvas to image
    #     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #     img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
    #     bridge = CvBridge()
    #     img_msg = Image()
    #     img_msg.image = bridge.cv2_to_imgmsg(img)
        
    #     self.network_publisher.publish(img_msg)
    #     plt.close(fig)

    def get_schedule(self) -> Dict[str, str]:
        # nodes = list(self.executor_clients.keys())
        # return {
        #     task: random.choice(nodes)
        #     for task in self.graph.task_names
        # }

        num_machines = len(self.all_nodes)
        num_tasks = len(self.graph.task_names)

        task_graph_ids = {
            task: i
            for i, task in enumerate(self.graph.task_names)
        }

        task_graph = {
            task_graph_ids[task.name]: [
                task_graph_ids[dep.name]
                for dep in deps
            ]
            for task, deps in self.graph.task_deps.items()
        }

        comm = torch.Tensor([
            [
                0 if i == j else 1 / self.bandwidths.get((self.all_nodes[i], self.all_nodes[j]), 1e9)
                for j in range(num_machines)
            ]
            for i in range(num_machines)
        ])

        schedule = find_schedule(
            num_of_all_machines=num_machines,
            num_node=num_tasks,
            input_given_speed=torch.ones(1, num_machines),
            input_given_comm=comm,
            input_given_comp=torch.ones(1,num_tasks),
            input_given_task_graph=task_graph
        )

        reverse_task_graph_ids = {v: k for k, v in task_graph_ids.items()}
        return {
            reverse_task_graph_ids[task_i]: self.all_nodes[node_i.item()]
            for task_i, node_i in enumerate(schedule)
        }
        
        
    def execute(self) -> Any:
        self.get_logger().info(f"\nSTARTING NEW EXECUTION")
        schedule = self.get_schedule()
        self.get_logger().info(f"SCHEDULE: {pformat(schedule)}")
        message = json.dumps({
            "execution_id": uuid4().hex,
            "data": {},
            "task_graph": self.graph.summary(),
            "schedule": schedule
        })

        futures = []
        for task in self.graph.start_tasks():
            node = schedule[task]
            req = Executor.Request()
            req.input = message
            self.get_logger().info(f"SENDING INITIAL TASK {task} TO {node}")
            futures.append((node, task, self.executor_clients[node].call_async(req)))
            
        return futures

    def execute_thread(self) -> None:
        while True:
            start = time.time()
            futures = self.execute()
            finished_futures = set()
            while len(finished_futures) < len(futures):
                for node, task, future in futures:
                    if future.done():
                        try:
                            response = future.result()
                        except Exception as e:
                            self.get_logger().error('Service call failed %r' % (e,))
                        else:
                            self.get_logger().info(f'RES FROM {node}: {response.output}')
                            finished_futures.add((node, task))
            time.sleep(max(0, self.interval - (time.time() - start)))

def main(args=None):
    rclpy.init(args=args)

    all_nodes = os.environ["ALL_NODES"].split(",")

    gcn_sched = Scheduler(
        nodes=all_nodes,
        graph=get_graph(),
        interval=10
    )
    thread = Thread(target=gcn_sched.execute_thread)
    thread.start()
    
    try:
        rclpy.spin(gcn_sched)
    finally:
        thread.join()
        gcn_sched.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
