#!/usr/bin/env python3
"""description_launch.py — 在 RViz 里看模型 + 用滑块摆姿势

    ros2 launch arm_description description_launch.py              # 滑块手动摆
    ros2 launch arm_description description_launch.py use_gui:=false  # 关节全零位

这一步只用模型，不涉及 Gazebo 与 ros2_control。
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_path = FindPackageShare(package='arm_description').find('arm_description')
    urdf_model_path = os.path.join(pkg_path, 'urdf', 'arm.urdf.xacro')
    rviz_config_path = os.path.join(pkg_path, 'rviz', 'description.rviz')

    use_gui = LaunchConfiguration('use_gui')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_urdf_model = DeclareLaunchArgument(
        name='urdf_model', default_value=urdf_model_path,
        description='要加载的 xacro / urdf 绝对路径')

    declare_rviz_config = DeclareLaunchArgument(
        name='rviz_config_file', default_value=rviz_config_path,
        description='RViz 配置文件绝对路径')

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time', default_value='false',
        description='是否使用仿真时钟')

    declare_use_gui = DeclareLaunchArgument(
        name='use_gui', default_value='true',
        description='是否启动关节滑块窗口')

    # 没有滑块时，也得有人往 /joint_states 发全零，否则 TF 树不完整
    start_joint_state_publisher = Node(
        condition=UnlessCondition(use_gui),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
    )

    start_joint_state_publisher_gui = Node(
        condition=IfCondition(use_gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    start_robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(pkg_path, 'launch', 'robot_state_publisher_launch.py')]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'urdf_model': urdf_model,
        }.items(),
    )

    start_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
    )

    return LaunchDescription([
        declare_urdf_model,
        declare_rviz_config,
        declare_use_sim_time,
        declare_use_gui,
        start_joint_state_publisher,
        start_joint_state_publisher_gui,
        start_robot_state_publisher,
        start_rviz,
    ])
