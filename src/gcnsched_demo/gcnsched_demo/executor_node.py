from interfaces.srv import Executor
import rclpy
from rclpy.node import Node, Client

import json
from task_graph import TaskGraph, deserialize, get_graph
from typing import Dict, List, Set, Generator, Tuple, Any
from queue import Queue
from threading import Thread


class ExecutorNode(Node):
    def __init__(self, 
                 name: str, 
                 graph: TaskGraph,
                 other_nodes: List[str]) -> None:
        super().__init__(name)
        self.name = name
        self.graph = graph
        self.data: Dict[str, str] = {}
        self.execution_history: Dict[str, Set] = {}
        self.queue = Queue()

        self.srv = self.create_service(
            Executor, f'{name}/executor', 
            self.executor_callback
        )

        self.clients: Dict[str, Client] = {}
        for other_node in other_nodes:
            self.clients[other_node] = self.create_client(Executor, f'/{other_node}/executor')

        thread = Thread(target=self.proccessing_thread)
        thread.start()

    def executor_callback(self, request, response) -> str:
        self.queue.put(request.input)
        response.output = "ack"
        return response

    def proccessing_thread(self) -> None:
        while True:
            msg = self.queue.get()
            for next_node, out_msg in self.process_message(msg):
                req = Executor.Request()
                req.input = out_msg
                res = self.clients[next_node].call(req)
                print(f"Acknowledgment from {next_node}: ", res.output)

    def process_message(self, msg_str: str) -> Generator[Tuple[str, str], None, None]:
        msg: Dict[str, Any] = json.loads(msg_str)
        execution_id: str = msg["execution_id"]

        # Update data
        data: Dict[str, str] = msg["data"]
        self.data.setdefault(execution_id, {})
        for task_name, data in data.items():
            self.data.setdefault(execution_id, {})
            self.data[execution_id][task_name] = data

        # Get tasks to execute on this node
        self.execution_history.setdefault(execution_id, set())
        schedule: Dict[str, str] = msg["schedule"]
        task_graph: Dict[str, List[str]] = msg["task_graph"]
        tasks = [
            task_name for task_name, node_name in schedule.items() 
            if (
                node_name == self.name and # task should be executed on this node
                task_name not in self.execution_history[execution_id] and # task has not been executed yet
                not (set(task_graph[task_name]) - set(self.data[execution_id].keys())) # all data is ready
            )
        ]

        for task in tasks:
            print(f"EXECUTING {task} ON {self.name}")
            args = [self.data[execution_id][dep] for dep in task_graph[task]]
            task_output = self.graph.execute(task, *args)
            self.execution_history[execution_id].add(task)
            next_nodes = {
                schedule[other_task] for other_task, deps in task_graph.items()
                if task in deps
            }
            if not next_nodes:
                print(f"OUTPUT: {task}: {deserialize(task_output)}")
            for next_node in next_nodes:
                msg = json.dumps(
                    {
                        "execution_id": execution_id,
                        "data": {task: task_output},
                        "task_graph": task_graph,
                        "schedule": schedule
                    },
                    indent=2
                )
                if self.name == next_node:
                    yield from self.process_message(msg) 
                else:
                    yield next_node, msg


def main(args=None):
    rclpy.init(args=args)

    ALL_NODES = ["node_1", "node_2", "node_3"]
    executor = ExecutorNode(
        name=args.name,
        graph=get_graph(),
        other_nodes=[node for node in ALL_NODES if node != args.name]
    )

    while rclpy.ok():
        rclpy.spin_once(executor)

    executor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
