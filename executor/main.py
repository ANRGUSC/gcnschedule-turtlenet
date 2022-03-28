from typing import Dict, Iterable, List, Set, Any, Generator, Tuple
from uuid import uuid4

import numpy as np
from task_graph import TaskGraph, deserialize, serialize
import json

class Node:
    def __init__(self, name: str, graph: TaskGraph) -> None:
        self.name = name
        self.graph = graph
        self.data: Dict[str, str] = {}
        self.execution_history: Dict[str, Set] = {}

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

class Network:
    def __init__(self, nodes: Iterable[Node]) -> None:
        self.nodes = {node.name: node for node in nodes}

    def send(self, src_node_name: str, dst_node_name: str, message: str) -> None:
        print(f"SENDING DATA FROM {src_node_name} -> {dst_node_name}")
        for next_node, output in self.nodes[dst_node_name].process_message(message):
            self.send(dst_node_name, next_node, output)

    def execute(self, graph: TaskGraph, schedule: Dict[str, str]) -> Any:
        message = json.dumps({
            "execution_id": uuid4().hex,
            "data": {},
            "task_graph": graph.summary(),
            "schedule": schedule
        })

        for task in graph.start_tasks():
            node = schedule[task]
            for next_node, output in self.nodes[node].process_message(message):
                self.send(node, next_node, output)

def get_graph():
    graph = TaskGraph()
    @graph.task()
    def generate_data() -> np.array:
        return np.zeros(100)

    @graph.task(generate_data)
    def add_noise(arr: np.ndarray) -> np.array:
        return arr + np.random.random(arr.shape)

    @graph.task(add_noise)
    def mean(arr: np.ndarray) -> float:
        return np.mean(arr)
        
    @graph.task(add_noise)
    def min(arr: np.ndarray) -> float:
        return np.min(arr)

    @graph.task(add_noise)
    def max(arr: np.ndarray) -> float:
        return np.max(arr)

    @graph.task(min, max)
    def midpoint(arr_min: float, arr_max: float) -> float:
        return (arr_min + arr_max) / 2

    return graph

def main():
    graph = get_graph()
    network = Network(
        nodes=[
            Node(f"node_{i}", graph=graph) 
            for i in range(1, 5)
        ]
    )

    network.execute(graph, schedule={
        "generate_data": "node_1",
        "add_noise": "node_1",
        "mean": "node_2",
        "min": "node_1",
        "max": "node_2",
        "midpoint": "node_3",
    })

if __name__ == "__main__":
    main()