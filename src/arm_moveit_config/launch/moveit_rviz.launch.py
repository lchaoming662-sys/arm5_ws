#!/usr/bin/env python3
"""moveit_rviz.launch.py — 只起 MoveIt 版 RViz（MotionPlanning 面板）

    ros2 launch arm_moveit_config moveit_rviz.launch.py

RViz 要能显示规划场景，必须拿到三份参数：
    robot_description           模型
    robot_description_semantic  SRDF（规划组定义）
    robot_description_kinematics IK 求解器（含 position_only_ik）
少了任意一份，MotionPlanning 面板会报错或直接不显示。
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("o5_2_arm", package_name="arm_moveit_config")
        .to_moveit_configs()
    )

    rviz_config = os.path.join(
        moveit_config.package_path, "config", "moveit.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="跟着 Gazebo 的 /clock 走时间（仿真时必须 true）"),
        DeclareLaunchArgument(
            "rviz_config", default_value=rviz_config,
            description="RViz 配置文件路径"),

        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
