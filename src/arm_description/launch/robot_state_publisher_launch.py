#!/usr/bin/env python3
"""robot_state_publisher_launch.py — 只起 robot_state_publisher

被 description_launch.py 复用；单独跑也可以，用来检查模型与 TF 树。

    ros2 launch arm_description robot_state_publisher_launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_path = FindPackageShare(package='arm_description').find('arm_description')
    default_urdf = os.path.join(pkg_path, 'urdf', 'arm.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')
    urdf_model = LaunchConfiguration('urdf_model')

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='false',
        description='是否使用仿真时钟（Gazebo 跑起来时设 true）')

    declare_urdf_model = DeclareLaunchArgument(
        name='urdf_model',
        default_value=default_urdf,
        description='要加载的 xacro / urdf 绝对路径')

    # xacro 展开结果必须是字符串，用 ParameterValue 显式声明类型，
    # 否则 ROS 2 会把它当成一个参数文件路径去解析。
    robot_description = {
        'robot_description': ParameterValue(Command(['xacro ', urdf_model]), value_type=str),
        'use_sim_time': use_sim_time,
    }

    start_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_urdf_model,
        start_robot_state_publisher,
    ])
