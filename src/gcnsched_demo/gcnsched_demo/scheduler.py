from pprint import pprint
import random
import time
from typing import Dict, List, Any
import rclpy
from rclpy.node import Node, Client

from interfaces.srv import Executor
import json
from uuid import uuid4

from .task_graph import TaskGraph, get_graph

class Scheduler(Node):
    def __init__(self,
                 nodes: List[str],
                 graph: TaskGraph,
                 interval: int) -> None:
        super().__init__('scheduler')
        print("INIT")
        self.graph = graph
        self.interval = interval

        self.executor_clients: Dict[str, Client] = {}
        for node in nodes:
            cli = self.create_client(
                Executor, f'/{node}/executor'
            )
            self.executor_clients[node] = cli
            while not cli.wait_for_service(timeout_sec=1.0):
                print(f'service /{node}/executor not available, waiting again...')

    def get_schedule(self) -> Dict[str, str]:
        nodes = list(self.executor_clients.keys())
        return {
            task: random.choice(nodes)
            for task in self.graph.task_names
        }

    def execute(self) -> Any:
        print(f"EXECUTING")
        schedule = self.get_schedule()
        pprint(schedule)
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
            print(f"SENDING {task} TO {node}")
            futures.append((node, task, self.executor_clients[node].call_async(req)))

        return futures

    def spin_execute(self) -> None:
        while True:
            start = time.time()
            futures = self.execute()
            finished_futures = set()
            while rclpy.ok():
                rclpy.spin_once(self)
                if len(finished_futures) == len(futures):
                    break
                for node, task, future in futures:
                    if future.done():
                        try:
                            response = future.result()
                        except Exception as e:
                            print('Service call failed %r' % (e,))
                        else:
                            print(f'RES FROM {node}: {response.output}')
                            finished_futures.add((node, task))
            time.sleep(max(0, self.interval - (time.time() - start)))

def main(args=None):
    rclpy.init(args=args)

    gcn_sched = Scheduler(
        nodes=["executor_1", "executor_2", "executor_3"],
        graph=get_graph(),
        interval=10
    )
    gcn_sched.spin_execute()

    gcn_sched.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
