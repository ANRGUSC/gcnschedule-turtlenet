from launch import LaunchDescription
from launch_ros.actions import Node

node_name = 'node2'
other_nodes = ['node1'] #'node3', 'node4']

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gcnsched_demo',
            node_executable='bandwidth',
            node_name='bandwidth',
            node_namespace='node2',
            output='screen',
            parameters=[
                {'name': node_name, 'other_nodes': other_nodes}
            ]
        )
    ])