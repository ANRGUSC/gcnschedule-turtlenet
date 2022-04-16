from asyncio import Future
from functools import partial
from pprint import pformat
from re import I
from threading import Thread
import time
import traceback
from typing import Dict, List, Tuple

import torch
from heft.core import schedule as schedule_dag

import rclpy
from rclpy.node import Node, Client, Publisher
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Image

from interfaces.srv import Executor
import json
from uuid import uuid4
from itertools import product

from gcnsched.ready_to_use import find_schedule
from .task_graph import TaskGraph, get_graph

from copy import deepcopy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from cv_bridge import CvBridge

class Scheduler(Node):
    def __init__(self,
                 nodes: List[str],
                 graph: TaskGraph) -> None:
        super().__init__('scheduler')
        #getting parameters from the launch file
        self.declare_parameter('nodes', [])
        self.declare_parameter('interval', 10)
        self.declare_parameter('scheduler', "heft")
        nodes = self.get_parameter('nodes').get_parameter_value().string_array_value
        self.scheduler = self.get_parameter('scheduler').get_parameter_value().string_value
        self.interval = self.get_parameter('interval').get_parameter_value().integer_value

        self.get_logger().info("SCHEDULER INIT")
        self.graph = graph
        self.all_nodes = nodes

        self.current_schedule = {}
        self.start_times = {}

        self.executor_clients: Dict[str, Client] = {}
        for node in nodes:
            cli = self.create_client(
                Executor, f'/{node}/executor'
            )
            self.executor_clients[node] = cli
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(f'service /{node}/executor not available, waiting again...')

        self.bandwidths: Dict[Tuple[str, str], float] = {}
        for src, dst in product(nodes, nodes):
            self.create_subscription(
                Float64, f"/{src}/{dst}/bandwidth",
                partial(self.bandwidth_callback, src, dst)
            )

        self.create_subscription(String, "/output", self.output_callback)
        self.makespan_publisher = self.create_publisher(Float64, "/makespan", 10)

        self.graph_publisher: Publisher = self.create_publisher(Image, "/taskgraph")
        self.create_timer(2, self.draw_task_graph)

        self.current_tasks: Dict[str, str] = {}
        for node in nodes:
            self.create_subscription(
                String, f"/{node}/current_task",
                partial(self.current_task_callback, node)
            )

    def get_bandwidth(self, n1: str, n2: str) -> float:
        now = time.time()
        ptime_1, bandwidth_1 = self.bandwidths.get((n1,n2), (0, 0))
        ptime_2, bandwidth_2 = self.bandwidths.get((n2,n1), (0, 0))
        if ptime_1 > ptime_2:
            bandwidth = bandwidth_1 if (now - ptime_1) < 10 else 0
        else:
            bandwidth = bandwidth_2 if (now - ptime_2) < 10 else 0

        return bandwidth + 1e-9

    def current_task_callback(self, node: str, msg: String) -> None:
        self.current_tasks[node] = msg.data

    def output_callback(self, msg: String) -> None:
        try:
            msg_data = json.loads(msg.data)
            makespan = Float64()
            makespan.data = time.time() - self.start_times[msg_data["execution_id"]]
            self.makespan_publisher.publish(makespan)
        except:
            self.get_logger().error(traceback.format_exc())

    def bandwidth_callback(self, src: str, dst: str, msg: Float64) -> None:
        self.bandwidths[(src, dst)] = time.time(), msg.data

    def get_schedule(self) -> Dict[str, str]:
        try:
            num_machines = len(self.all_nodes)
            num_tasks = len(self.graph.task_names)

            task_graph_ids = {
                task: i
                for i, task in enumerate(self.graph.task_names)
            }

            reverse_task_graph_ids = {v: k for k, v in task_graph_ids.items()}

            task_graph = {
                task_graph_ids[task.name]: [
                    task_graph_ids[dep.name]
                    for dep in deps
                ]
                for task, deps in self.graph.task_deps.items()
            }

            comm = torch.Tensor([
                [
                    0 if i == j else min(1e9, 1 / self.get_bandwidth(self.all_nodes[i], self.all_nodes[j]))
                    for j in range(num_machines)
                ]
                for i in range(num_machines)
            ])

            comp = torch.Tensor([
                [
                    self.graph.task_names[reverse_task_graph_ids[i]].cost
                    for i in range(num_tasks)
                ]
            ])

            task_graph_forward: Dict[int, List[int]] = {}
            for node_name, node_deps in task_graph.items():
                for node_dep in node_deps:
                    task_graph_forward.setdefault(node_dep, [])
                    if node_name not in task_graph_forward[node_dep]:
                        task_graph_forward[node_dep].append(node_name)

            

            # sched = find_schedule(
            #     num_of_all_machines=num_machines,
            #     num_node=num_tasks,
            #     input_given_speed=torch.ones(1, num_machines),
            #     input_given_comm=comm,
            #     input_given_comp=comp,
            #     input_given_task_graph=task_graph_forward
            # )

            # return {
            #     reverse_task_graph_ids[task_i]: self.all_nodes[node_i.item()]
            #     for task_i, node_i in enumerate(sched)
            # }

            _, jobson = schedule_dag(
                task_graph_forward,
                agents=self.all_nodes,
                compcost=lambda task_id, agent: self.graph.task_names[reverse_task_graph_ids[task_id]].cost,
                commcost=lambda t1, t2, a1, a2: 0 if a1 == a2 else 1 / self.get_bandwidth(a1, a2)
            )

            return {
                reverse_task_graph_ids[task_id]: agent
                for task_id, agent in jobson.items()
            }

            
        except:
            self.get_logger().error(traceback.format_exc())
        finally:
            self.get_logger().debug("exiting get schedule function")

    def execute(self) -> Tuple[str, List[Future]]:
        self.get_logger().info(f"\nSTARTING NEW EXECUTION")
        schedule = self.get_schedule()
        self.current_schedule = deepcopy(schedule)
        self.get_logger().info(f"SCHEDULE: {pformat(schedule)}")
        execution_id = uuid4().hex
        message = json.dumps({
            "execution_id": execution_id,
            "data": {},
            "task_graph": self.graph.summary(),
            "schedule": schedule
        })

        self.start_times[execution_id] = time.time()
        futures = []
        for task in self.graph.start_tasks():
            node = schedule[task]
            req = Executor.Request()
            req.input = message
            self.get_logger().info(f"SENDING INITIAL TASK {task} TO {node}")
            futures.append((node, task, self.executor_clients[node].call_async(req)))

        return futures

    def execute_thread(self) -> None:
        try:
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
                        else:
                            print(f"Future not done for {node} executing {task}")
                    time.sleep(self.interval / 10)
                time.sleep(max(0, self.interval - (time.time() - start)))
        except:
            self.get_logger().error(traceback.format_exc())
        finally:
            self.get_logger().error("EXITING SCHEDULER")

    def draw_task_graph(self) -> None:
        try:
            start = time.time()
            self.get_logger().info("Drawing taskgraph")

            graph = nx.DiGraph()

            for task in self.graph.task_deps:
                for dep in self.graph.task_deps[task]:
                    graph.add_edge(dep.name,task.name)

            self.get_logger().info(str(list(graph.nodes)))
            types = {
                t: i for i, t in enumerate(self.all_nodes)
            }
            self.get_logger().info("types "+str(types))
            self.get_logger().info("nodes "+str(self.all_nodes))
            
            current_schedule = deepcopy(self.current_schedule)
            {node: task for task, node in current_schedule.items()}
            node_color = [
                types[current_schedule[node]] for node in graph.nodes
            ]
            self.get_logger().info("color "+str(node_color))
            cmap = cm.get_cmap('gist_rainbow', len(self.all_nodes))
            labels_dict = dict([(node_name, node_name.split("_")[0][:7]) for node_name in graph.nodes])

            pos = nx.nx_agraph.graphviz_layout(graph,prog='dot')
            fig = plt.figure()
            nx.draw(
                graph, pos, edge_color='black', width=1, linewidths=1,
                node_size=200, node_color=node_color, alpha=0.8,
                labels=labels_dict,
                with_labels = False, 
                font_weight = 'bold',
                font_size=8,
                cmap=cmap,vmin=0,vmax=len(self.all_nodes)
            )
            color_lines = [mpatches.Patch(color=cmap(types[t]), label=t) for t in types.keys()]
            legend = plt.legend(handles=color_lines, loc='best')
            # nx.draw_networkx_edge_labels(
            #     graph, pos,
            #     edge_labels=bandwidths,
            #     font_color='red'
            # )

            fig.canvas.draw()

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # self.get_logger().info(str(img))

            bridge = CvBridge()
            img_msg = Image()
            # img_msg.header.stamp = self.get_clock().now()
            img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")

            self.graph_publisher.publish(img_msg)
            self.get_logger().info(f"publishing task graph image for visualization {time.time()-start} seconds")
            plt.close(fig)
        except:
            self.get_logger().error(traceback.format_exc())
        finally:
            self.get_logger().info(f"leaving task graph drawer")

def main(args=None):
    rclpy.init(args=args)

    all_nodes = []

    gcn_sched = Scheduler(
        nodes=all_nodes,
        graph=get_graph()
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
