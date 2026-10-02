#!/usr/bin/env python3
"""demo.launch.py — move_group + MoveIt 版 RViz（两个一起起）

**前提：仿真已经在跑**（另一个终端里 `ros2 launch arm_gazebo gz_launch.py` 还开着）。
本文件按仿真已经提供 robot_state_publisher 与 controller_manager 来设计，
所以不会再起它们（理由见 move_group.launch.py 的说明）。

完整启动顺序：

    终端 A:  ros2 launch arm_gazebo gz_launch.py          # 仿真 + 控制器
    终端 B:  ros2 launch arm_moveit_config demo.launch.py # MoveIt + RViz

然后 RViz 的 MotionPlanning 面板里：
    ① Planning 标签页 → Start State 选 <current>，Goal State 选 <random valid> 或拖交互标记
    ② 点 Plan 看规划出的轨迹
    ③ 点 Execute 让真机（这里是仿真）动起来
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_launch = FindPackageShare("arm_moveit_config")

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_launch, "launch", "move_group.launch.py"])),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_launch, "launch", "moveit_rviz.launch.py"])),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="跟着 Gazebo 的 /clock 走时间（仿真时必须 true）"),
        DeclareLaunchArgument(
            "use_rviz", default_value="true",
            description="是否同时起 MoveIt 版 RViz"),

        move_group,
        rviz,
    ])
