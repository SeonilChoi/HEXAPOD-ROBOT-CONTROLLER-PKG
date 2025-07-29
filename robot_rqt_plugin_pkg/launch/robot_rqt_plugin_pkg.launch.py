from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    robot_rqt_plugin = Node(
        package='robot_rqt_plugin_pkg',
        executable='run_rqt',
        name='robot_rqt_plugin',
        namespace='canopen',
        output='screen'
    )

    return LaunchDescription([
        robot_rqt_plugin
    ])
