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

    # capabilities 必须显式列全：这个参数一旦设置就会【替换】move_group 的
    # 默认列表，而不是追加。不设的时候实测只加载了 5 个（ApplyPlanningScene /
    # ExecuteTrajectory / GetPlanningScene / MotionPlan / QueryPlanners），
    # 连 MoveGroupMoveAction 都没有。
    #
    # 末尾的 ExecuteTaskSolutionCapability 是 MTC 执行所必需的：
    # MTC 的 Task.execute() 文档写的是「Send given solution to move_group node
    # for execution」，它走的就是这个 capability 提供的 action。
    # 缺了它，MTC 规划能成功、执行却直接返回 FAILURE(99999)，
    # 而且 move_group 侧几乎不打印任何东西 —— 很难从现象反推原因。
    capabilities = " ".join([
        "move_group/ApplyPlanningSceneService",
        "move_group/ClearOctomapService",
        "move_group/MoveGroupCartesianPathService",
        "move_group/MoveGroupExecuteTrajectoryAction",
        "move_group/MoveGroupGetPlanningSceneService",
        "move_group/MoveGroupKinematicsService",
        "move_group/MoveGroupMoveAction",
        "move_group/MoveGroupPickPlaceAction",
        "move_group/MoveGroupPlanService",
        "move_group/MoveGroupQueryPlannersService",
        "move_group/MoveGroupStateValidationService",
        "move_group/ExecuteTaskSolutionCapability",
    ])

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
                {"capabilities": capabilities},
            ],
        ),
    ])
