from base64 import b64encode, b64decode
from typing import Any, Callable, Dict, List
import pickle

import inspect
import time
from wfcommons.common.workflow import Workflow
from wfcommons.wfchef.recipes.cycles.recipe import CyclesRecipe
from wfcommons.wfchef.recipes.montage import MontageRecipe
from wfcommons.wfchef.recipes.seismology import SeismologyRecipe
from wfcommons.wfchef.recipes.blast import BlastRecipe
from wfcommons.wfchef.recipes.bwa import BwaRecipe
from wfcommons.wfchef.recipes.epigenomics import EpigenomicsRecipe
from wfcommons.wfchef.recipes.srasearch import SrasearchRecipe
from wfcommons.wfchef.recipes.genome import GenomeRecipe
from wfcommons.wfchef.recipes.soykb import SoykbRecipe
from wfcommons.wfchef.utils import draw

import pathlib
import json

RECIPES = {
    "montage": MontageRecipe,
    "cycles": CyclesRecipe,
    "seismology": SeismologyRecipe,
    "blast": BlastRecipe,
    "bwa": BwaRecipe,
    "epigenomics": EpigenomicsRecipe,
    "srasearch": SrasearchRecipe,
    "genome": GenomeRecipe,
    "soykb": SoykbRecipe
}

RECIPE = "epigenomics"
NUM_TASKS = 1000

def deserialize(text: str) -> Any:
    return pickle.loads(b64decode(text))

def serialize(obj: Any) -> str:
    return b64encode(pickle.dumps(obj)).decode("utf-8")

class Task:
    def __init__(self, name: str, call: Callable, cost: float = 0.0) -> None:
        self.name = name
        self.call = call
        self.cost = cost

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

class TaskGraph:
    def __init__(self) -> None:
        self.task_deps: Dict[Task, List[Task]] = {}
        self.task_names: Dict[str, Task] = {}

    def execute(self, name: str, *args, **kwargs) -> Any:
        return self.task_names[name](*args, **kwargs)

    def dependencies(self, name: str) -> List[str]:
        return [
            task.name for task in
            self.task_deps[self.task_names[name]]
        ]

    def task(self, *deps: Task) -> Callable:
        def _task(fun: Callable) -> Any:
            return self.add_task(fun, *deps)
        return _task

    def _add_task(self, task: Task, *deps: Task) -> None:
        self.task_deps[task] = deps
        self.task_names[task.name] = task
        
    def add_task(self, fun: Callable, *deps: Task) -> Task:
        def _fun(*args, **kwargs) -> Any:
            args = [deserialize(arg) for arg in args]
            kwargs = {
                key: deserialize(value) 
                for key, value in kwargs.items()
            }
            return serialize(fun(*args, **kwargs))
        task = Task(fun.__name__, _fun)
        self._add_task(task, deps)
        return task

    def __str__(self) -> str:
        return "\n".join([
            f"{task.name} <- [{', '.join([dep.name for dep in deps])}]"
            for task, deps in self.task_deps.items()
        ])

    def start_tasks(self) -> List[str]:
        return [
            task_name for task_name, task in self.task_names.items()
            if not self.task_deps[task]
        ]

    def end_tasks(self) -> List[str]:
        source_tasks = {
            dep.name for _, task in self.task_names.items() 
            for dep in self.task_deps[task]
        }
        return list(set(self.task_names.keys()) - source_tasks)

    def summary(self) -> Dict[str, List[str]]:
        return {
            task.name: [dep.name for dep in deps]
            for task, deps in self.task_deps.items()
        }



thisdir = pathlib.Path(__file__).resolve().parent
task_stats_path = pathlib.Path(inspect.getfile(RECIPES[RECIPE])).parent.joinpath("task_type_stats.json")
task_stats = json.loads(pathlib.Path(task_stats_path).read_text())

def get_graph() -> TaskGraph:
    recipe = RECIPES[RECIPE](num_tasks=NUM_TASKS) 
    workflow: Workflow = recipe.build_workflow("my_workflow")

    # draw(workflow, save=thisdir.joinpath("graph"))

    task_functions = {}
    tasks = {}
    for node in workflow.nodes:
        print((workflow.nodes[node]["task"].category))
        task_type = workflow.nodes[node]["task"].category
        stats = task_stats[task_type]
        runtime = (stats["runtime"]["max"] - stats["runtime"]["min"])/2

        if task_type not in tasks:
            def run(*args, **kwargs) -> str:
                time.sleep(runtime)
                return "Nothing"

            task_functions[task_type] = run
        
        tasks[node] = Task(node, run, cost=runtime)

    src = Task("SRC", lambda *args, **kwargs: "SRC OUTPUT")
    dst = Task("DST", lambda *args, **kwargs: "DST OUTPUT")

    task_graph = TaskGraph()
    end_tasks = []
    
    task_graph._add_task(src) 
    for name, task in tasks.items():
        if workflow.in_degree(name) == 0:
            task_graph._add_task(task, src)
        else:
            task_graph._add_task(task, *[tasks[dep] for dep, _ in workflow.in_edges(name)])

        if workflow.out_degree(name) == 0:
            end_tasks.append(task)

    task_graph._add_task(dst, *end_tasks) 

    return task_graph
    

if __name__ == "__main__":
    print(get_graph())