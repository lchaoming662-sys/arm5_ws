#!/usr/bin/env python3
"""move_group.launch.py — 只起 MoveIt 的规划节点

设计前提：**机械臂已经在别的地方跑起来了**（本项目是 Gazebo Fortress 里，
见 `ros2 launch arm_gazebo gz_launch.py`）。

所以本文件【只】起 move_group，**不**起：
    · 第二个 robot_state_publisher   （仿真那边已有，同名会互相抢）
    · ros2_control_node              （控制器由 gz_ros2_control 插件在仿真进程内提供）
    · controller spawner             （仿真那边已经激活过了）

不要改用 moveit_configs_utils 的 `generate_demo_launch()`：
   它会额外起一个 ros2_control_node + 一串 spawner，
   和已在跑的 Gazebo 撞车，表现为 spawner 报 Failed loading controller。

    ros2 launch arm_moveit_config move_group.launch.py
"""

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

    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="跟着 Gazebo 的 /clock 走时间（仿真时必须 true）"),

        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
