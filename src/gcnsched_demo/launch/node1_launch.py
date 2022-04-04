from launch import LaunchDescription
from launch_ros.actions import Node

node_name = 'node1'
other_nodes = ['node2'] #'node3', 'node4']

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gcnsched_demo',
            node_executable='bandwidth',
            node_name='bandwidth',
            node_namespace='node1',
            output='screen',
            parameters=[
                {'name': node_name, 'other_nodes': other_nodes}
            ]
        )
    ])