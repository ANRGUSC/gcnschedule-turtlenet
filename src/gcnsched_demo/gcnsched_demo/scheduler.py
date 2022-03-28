from typing import Dict, List, Any
from urllib.request import Request
from gcnsched_demo.gcnsched_demo.executor_node import ExecutorNode
import rclpy
from rclpy.node import Node, Client

from std_msgs.msg import String
from interfaces.srv import Executor
from interfaces.msg import Num # custom msg type
from task_graph import TaskGraph, get_graph
import json
from uuid import uuid4

class Scheduler(Node):

    def __init__(self, 
                 nodes: List[str], 
                 graph: TaskGraph,
                 schedule: Dict[str, str]) -> None:
        print("INIT")
        super().__init__('scheduler')
        self.publisher_ = self.create_publisher(Num, 'scheduler_status', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        self.graph = graph
        self.schedule = schedule

        self.clients: Dict[str, Client] = {}
        for node in nodes:
            self.clients[node] = self.create_client(Executor, f'/{node}/executor')

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
            res = self.clients[node].call(req)
            print(f"Acknowledgement: {res.output}")

    # def timer_callback(self):
    #     msg = Num()
    #     msg.num = self.i
    #     self.publisher_.publish(msg)
    #     self.get_logger().info('Publishing scheduler status: "%d"' % msg.num)
    #     self.i += 1


def main(args=None):
    print("HERE")
    rclpy.init(args=args)

    gcn_sched = Scheduler(
        nodes=["node_1", "node_2", "node_3"],
        graph=get_graph(),
        schedule={
            "generate_data": "node_1",
            "add_noise": "node_1",
            "mean": "node_2",
            "min": "node_1",
            "max": "node_2",
            "midpoint": "node_3",
        }
    )
    gcn_sched.execute()

    rclpy.spin(gcn_sched)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gcn_sched.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
