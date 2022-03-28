from typing import Dict, List, Set, Any
from task_graph import TaskGraph, graph, deserialize, serialize
import json

class Node:
    def __init__(self, name: str, graph: TaskGraph) -> None:
        self.name = name
        self.graph = graph
        self.data: Dict[str, str] = {}
        self.execution_history: Dict[str, Set] = {}

    def process_message(self, msg_str: str):
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
            args = [self.data[execution_id][dep] for dep in task_graph[task]]
            task_output = self.graph.execute(task, *args)
            self.execution_history[execution_id].add(task)
            next_nodes = {
                schedule[other_task] for other_task, deps in task_graph.items()
                if task in deps
            }
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
                    self.process_message(msg) 
                else:
                    print(f"SEND {self.name} -> {next_node}:\n", msg)
            

        print(graph.dependencies("add_noise"))

def main():
    node = Node("node_1", graph)
    message = {
        "execution_id": "asdf",
        "data": {},
        "task_graph": graph.summary(),
        "schedule": {
            "generate_data": "node_1",
            "mean": "node_1",
            "min": "node_2",
            "max": "node_2",
            "midpoint": "node_3",
            "add_noise": "node_1"
        }
    }
    node.process_message(json.dumps(message))

if __name__ == "__main__":
    main()