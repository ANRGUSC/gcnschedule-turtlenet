import time
from typing import Dict, List, Any
import rclpy
from rclpy.node import Node, Client

from std_msgs.msg import String
from interfaces.srv import Executor
from interfaces.msg import Num # custom msg type
import json
from uuid import uuid4

from .task_graph import TaskGraph, get_graph

class Scheduler(Node):

    def __init__(self, 
                 nodes: List[str], 
                 graph: TaskGraph,
                 schedule: Dict[str, str]) -> None:
        print("INIT")
        super().__init__('scheduler')
        self.publisher_ = self.create_publisher(Num, 'scheduler_status', 10)

        self.graph = graph
        self.schedule = schedule

        self.executor_clients: Dict[str, Client] = {}
        for node in nodes:
            self.executor_clients[node] = self.create_client(Executor, f'/{node}/executor')

    def execute(self) -> Any:
        print("EXECUTING")
        message = json.dumps({
            "execution_id": uuid4().hex,
            "data": {},
            "task_graph": self.graph.summary(),
            "schedule": self.schedule
        })

        for task in self.graph.start_tasks():
            node = self.schedule[task]
            
            req = Executor.Request()
            req.input = message
            res = self.executor_clients[node].call(req)

    # def timer_callback(self):
    #     msg = Num()
    #     msg.num = self.i
    #     self.publisher_.publish(msg)
    #     self.get_logger().info('Publishing scheduler status: "%d"' % msg.num)
    #     self.i += 1


def main(args=None):
    rclpy.init(args=args)

    gcn_sched = Scheduler(
        nodes=["executor_1", "executor_2", "executor_3"],
        graph=get_graph(),
        schedule={
            "generate_data": "executor_1",
            "add_noise": "executor_1",
            "mean": "executor_2",
            "min": "executor_1",
            "max": "executor_2",
            "midpoint": "executor_3",
        }
    )

    time.sleep(5)
    while True:
        gcn_sched.execute()
        print("Spinning")
        rclpy.spin_once(gcn_sched, timeout_sec=1)
        print("Done Spinning")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gcn_sched.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
