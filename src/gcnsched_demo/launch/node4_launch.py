from launch import LaunchDescription
from launch_ros.actions import Node

node_name = 'node4'
other_nodes = ['node1', 'node3', 'node2']

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gcnsched_demo',
            node_executable='bandwidth',
            node_name=node_name,
            node_namespace=node_name,
            output='screen',
            parameters=[
                {'name': node_name, 'other_nodes': other_nodes, }
            ]
        ),
        Node(
            package='gcnsched_demo',
            node_executable='executor',
            node_name=node_name,
            node_namespace=node_name,
            output='screen',
            parameters=[
                {'name': node_name, 'other_nodes': other_nodes, }
            ]
        )
    ])