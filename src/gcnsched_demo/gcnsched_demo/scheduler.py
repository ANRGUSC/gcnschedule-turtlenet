from functools import partial
from pprint import pformat
import random
from threading import Thread
import time
from typing import Dict, List, Any


import rclpy
from rclpy.node import Node, Client, Publisher
from std_msgs.msg import Float64
from sensor_msgs.msg import Image

from interfaces.srv import Executor
import json
from uuid import uuid4
import os
from itertools import product

from .task_graph import TaskGraph, get_graph

class Scheduler(Node):
    def __init__(self,
                 nodes: List[str],
                 graph: TaskGraph,
                 interval: int) -> None:
        super().__init__('scheduler')
        #getting parameters from the launch file
        self.declare_parameter('nodes', [])
        self.declare_parameter('interval', 10)
        nodes = self.get_parameter('nodes').get_parameter_value().string_array_value
        interval = self.get_parameter('interval').get_parameter_value().integer_value
        print(nodes)

        self.get_logger().info("SCHEDULER INIT")
        self.graph = graph
        self.interval = interval

        self.executor_clients: Dict[str, Client] = {}
        for node in nodes:
            cli = self.create_client(
                Executor, f'/{node}/executor'
            )
            self.executor_clients[node] = cli
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(f'service /{node}/executor not available, waiting again...')

        self.bandwidths: Dict[str, Dict[str, float]] = {}
        for src, dst in product(nodes, nodes):
            self.create_subscription(
                Float64, f"/{src}/{dst}/bandwidth",
                partial(self.bandwidth_callback, src, dst)
            )

    def bandwidth_callback(self, src: str, dst: str, msg: Float64) -> None:
        self.bandwidths[(src, dst)] = msg.data
        # print("BANDWIDTHS:", pformat(self.bandwidths))

    def get_schedule(self) -> Dict[str, str]:
        nodes = list(self.executor_clients.keys())
        return {
            task: random.choice(nodes)
            for task in self.graph.task_names
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
        print('execute_thread')
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

    all_nodes = []

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
