from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    hexapod_controller = Node(
        package='hexapod_rqt_plugin_pkg',
        executable='run_rqt',
        name='hexapod_controller',
        namespace='canopen',
        output='screen'
    )

    return LaunchDescription([
        hexapod_controller
    ])