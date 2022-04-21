from launch import LaunchDescription
from launch_ros.actions import Node

node_name = 'node3'
other_nodes = ['node1', 'node2', 'node4']
interval = 5.0

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='gcnsched_demo',
        #     node_executable='bandwidth',
        #     node_name='bandwidth',
        #     node_namespace=node_name,
        #     prefix=['stdbuf -o L'],
        #     output='screen',
        #     parameters=[
        #         {'name': node_name, 'other_nodes': other_nodes, }
        #     ],
        #     # arguments=[('__log_level:=debug')]
        # ),
        # Node(
        #     package='gcnsched_demo',
        #     node_executable='executor',
        #     node_name='executor',
        #     node_namespace=node_name,
        #     prefix=['stdbuf -o L'],
        #     output='screen',
        #     parameters=[
        #         {'name': node_name, 'other_nodes': other_nodes, }
        #     ]
        # )
        Node(
            package='gcnsched_demo',
            node_executable='ping',
            node_name='ping',
            node_namespace=node_name,
            prefix=['stdbuf -o L'],
            output='screen',
            parameters=[
                {'name': node_name, 'other_nodes': other_nodes, 'interval': 5.0 }
            ],
        )
    ])
