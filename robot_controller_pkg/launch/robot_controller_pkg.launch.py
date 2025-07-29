import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    robot_controller_pkg_path = get_package_share_directory('robot_controller_pkg')
    
    robot_controller_node = Node(
        package='robot_controller_pkg',
        executable='robot_controller_node',
        name='robot_controller_node',
        namespace='canopen',
        output='screen',
    )

    return LaunchDescription([
        robot_controller_node
    ])
