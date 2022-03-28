from base64 import b64encode, b64decode
from typing import Any, Callable, Dict, List
import pickle

import numpy as np

def deserialize(text: str) -> Any:
    return pickle.loads(b64decode(text))

def serialize(obj: Any) -> str:
    return b64encode(pickle.dumps(obj)).decode("utf-8")

class Task:
    def __init__(self, name: str, call: Callable) -> None:
        self.name = name
        self.call = call

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

class TaskGraph:
    def __init__(self) -> None:
        self.tasks: Dict[Task, List[Task]] = {}
        self.task_names: Dict[str, Task] = {}

    def execute(self, name: str, *args, **kwargs) -> Any:
        return self.task_names[name](*args, **kwargs)

    def dependencies(self, name: str) -> List[str]:
        return [
            task.name for task in
            self.tasks[self.task_names[name]]
        ]
    
    def task(self, *deps: Task) -> Callable:
        def _task(fun: Callable) -> Task:
            def _fun(*args, **kwargs) -> Any:
                args = [deserialize(arg) for arg in args]
                kwargs = {key: deserialize(value) for key, value in kwargs.items()}
                return serialize(fun(*args, **kwargs))
            task = Task(fun.__name__, _fun)
            self.tasks[task] = deps
            self.task_names[task.name] = task
            return task
        return _task

    def __str__(self) -> str:
        return "\n".join([
            f"{task.name} <- [{', '.join([dep.name for dep in deps])}]"
            for task, deps in self.tasks.items()
        ])

    def summary(self) -> Dict[str, List[str]]:
        return {
            task.name: [dep.name for dep in deps]
            for task, deps in self.tasks.items()
        }

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
