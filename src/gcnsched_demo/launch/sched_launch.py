from launch import LaunchDescription
from launch_ros.actions import Node

# node_name = 'node1'
# other_nodes = ['node2'] #'node3', 'node4']
nodes = ['node1', 'node3', 'node4']

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gcnsched_demo',
            node_executable='scheduler',
            # node_name=node_name,
            # node_namespace=node_name,
            prefix=['stdbuf -o L'],
            output='screen',
            parameters=[
                {'nodes': nodes }
            ]
        ),
        Node(
            package='gcnsched_demo',
            node_executable='visualizer',
            # node_name=node_name,
            # node_namespace=node_name,
            prefix=['stdbuf -o L'],
            output='screen',
            parameters=[
                {'nodes': nodes }
            ]
        )
    ])
