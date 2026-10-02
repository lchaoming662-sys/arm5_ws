#!/usr/bin/env python3
"""grasp_mtc.launch.py — 用 MoveIt Task Constructor 跑抓取流水线

这里**必须**用 launch 启动，不能直接 python 跑脚本：MTC 需要整套 MoveIt 参数
（robot_description / robot_description_semantic / robot_description_kinematics
/ joint_limits / planning_pipelines），而它们只有 launch 能注入 —— MTC 的 Python
绑定建在 rclcpp 上，rclcpp.Node 的 Python 接口只有一个 name 属性，没有声明参数
的途径。少任何一项都以很隐蔽的方式退化（实测）：

    · 缺 kinematics.yaml    -> "No kinematics solver instantiated for group 'arm'"
                              → 25 个候选全部 IK 失败，看着像"这臂够不着"
    · 缺 ompl_planning.yaml -> 悄悄退化成 CHOMP 规划器，不报错，
                              但规划的已经不是我们配的那套

前置：仿真与 move_group 已在跑（`ros2 launch arm_bringup bringup.launch.py`
一条命令即可）。

    ros2 launch arm_bringup grasp_mtc.launch.py
    ros2 launch arm_bringup grasp_mtc.launch.py display:=true   # 发到 RViz 的 MTC 面板
    ros2 launch arm_bringup grasp_mtc.launch.py execute:=true   # 规划完直接执行
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # 与 move_group.launch.py 用同一套组装方式：
    # MoveItConfigsBuilder 会顺着 arm_moveit_config 的 .setup_assistant
    # 找到 URDF 与各配置，一次性把该给 MoveIt 的参数都凑齐。
    moveit_config = (
        MoveItConfigsBuilder("o5_2_arm", package_name="arm_moveit_config")
        .to_moveit_configs()
    )

    display = LaunchConfiguration("display")
    execute = LaunchConfiguration("execute")
    probe = LaunchConfiguration("probe")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "display", default_value="false",
            description="把解发到 RViz 的 MTC 面板（Motion Planning Tasks）"),
        DeclareLaunchArgument(
            "execute", default_value="false",
            description="规划成功后直接执行（仿真/真机会动起来）"),
        DeclareLaunchArgument(
            "probe", default_value="false",
            description="诊断：只生成抓取候选并打印它们的位姿，不做 IK"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="跟着 Gazebo 的 /clock 走时间（仿真时必须 true）"),

        # 脚本用这几个环境变量决定走哪条分支
        SetEnvironmentVariable(name="MTC_DISPLAY", value=display),
        SetEnvironmentVariable(name="MTC_EXECUTE", value=execute),
        SetEnvironmentVariable(name="MTC_PROBE", value=probe),

        Node(
            package="arm_bringup",
            executable="grasp_pipeline_mtc.py",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
