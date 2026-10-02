#!/usr/bin/env python3
"""bringup.launch.py — 一条命令拉起整条链路

    仿真（Gazebo Fortress + robot_state_publisher + 桥 + 控制器） + MoveIt（move_group + RViz）

    ros2 launch arm_bringup bringup.launch.py                 # 全开（默认）
    ros2 launch arm_bringup bringup.launch.py gui:=false      # 不开 Gazebo 界面，跑得快点
    ros2 launch arm_bringup bringup.launch.py use_rviz:=false # 不起 RViz（只跑 move_group）
    ros2 launch arm_bringup bringup.launch.py use_moveit:=false  # 只要仿真 + 控制器

==============================================================================
它做的事 / 不做的事
==============================================================================
做：把下面两条本来要开两个终端分别敲的命令合成一条，
    并把 MoveIt 的启动**卡在控制器真正 active 之后**。

    终端 A:  ros2 launch arm_gazebo gz_launch.py
    终端 B:  ros2 launch arm_moveit_config demo.launch.py

不做（也**不能**做）：
    · 不再起一个 robot_state_publisher —— 仿真那一路已经有了，同名会互相抢
    · 不起 ros2_control_node —— 真控制器由 gz_ros2_control 插件在 Gazebo 进程内提供，
      外部再起会撞名（历史坑见 docs/踩坑记录.md）
    这两条是同名的老坑，改本文件时不要「顺手补上」。

==============================================================================
为什么中间夹一个门闩节点
==============================================================================
仿真那路的三个 spawner 是在 gz_launch.py 里用 OnProcessExit 一层层串的，
而且那是**被 include 进来**的动作，外层 launch 的 OnProcessExit 挂不到它
（target_action 只认本文件内定义的动作）。又不能用固定延时：
本机 RTF 仅 0.39，固定 30 s 真实时间只等于约 12 s 仿真时间，机器一忙就不够。

所以用 `arm_bringup/scripts/wait_for_controllers.py` 去**问状态**——
它轮询 /controller_manager/list_controllers，直到点名的那几个控制器
全部 active 就退出；本文件用 OnProcessExit 接住它再拉起 MoveIt。
等多久取决于真的好了没有，不猜。

注意：门闩超时（退出码 1）时本文件**照样**会启动 MoveIt。
这是故意的——MoveIt 起得来，用户能在日志里看到那条超时报错、
再去查 Gazebo 为什么没起来，比整条链路静默不动要好。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# 门闩要等的那几个控制器，必须与 arm_bringup/config/arm_controllers.yaml 里的一致。
# joint_state_broadcaster 排在最前：它是 MoveIt 拿到 /joint_states 的前提。
READY_CONTROLLERS = [
    'joint_state_broadcaster',
    'arm_controller',
    'gripper_controller',
]


def generate_launch_description():
    pkg_bringup = get_package_share_directory('arm_bringup')
    pkg_gz = get_package_share_directory('arm_gazebo')
    pkg_moveit = get_package_share_directory('arm_moveit_config')

    gui = LaunchConfiguration('gui')
    camera = LaunchConfiguration('camera')
    world = LaunchConfiguration('world')
    use_moveit = LaunchConfiguration('use_moveit')
    use_rviz = LaunchConfiguration('use_rviz')
    gate_timeout = LaunchConfiguration('gate_timeout')

    # ── ① 仿真这一路：模型 / 桥 / spawn / 三个控制器，全在 gz_launch.py 里 ──
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gz, 'launch', 'gz_launch.py')),
        launch_arguments={
            'gui': gui,
            'camera': camera,
            'world': world,
        }.items(),
    )

    # ── ② 门闩：等控制器真的 active（细节见文件头） ──
    gate = Node(
        package='arm_bringup',
        executable='wait_for_controllers.py',
        # --reactivate：全部就绪之后再让夹爪控制器重新申领一次命令接口。
        # 不做这一步，right_claw_joint 就完全不动（原因见该脚本的说明）。
        arguments=['--controllers'] + READY_CONTROLLERS
                  + ['--reactivate', 'gripper_controller']
                  + ['--timeout', gate_timeout],
        output='screen',
        condition=IfCondition(use_moveit),
    )

    # ── ③ MoveIt 这一路：move_group + RViz ──
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_moveit, 'launch', 'demo.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'use_rviz': use_rviz,
        }.items(),
    )

    after_gate = RegisterEventHandler(
        OnProcessExit(
            target_action=gate,
            on_exit=[
                LogInfo(msg='控制器已就绪 → 启动 move_group 与 RViz'),
                moveit,
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='是否打开 Gazebo 界面。false = 只跑 server，仿真更快'),
        DeclareLaunchArgument(
            'camera', default_value='true',
            description='是否加载腕部相机（含传感器与图片桥）。false 可提速'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_gz, 'worlds', 'arm_world.sdf'),
            description='要加载的 SDF 世界文件路径'),
        DeclareLaunchArgument(
            'use_moveit', default_value='true',
            description='是否在控制器就绪后启动 MoveIt（move_group + RViz）'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='MoveIt 起不起 RViz'),
        DeclareLaunchArgument(
            'gate_timeout', default_value='180',
            description='等控制器 active 的总超时（秒）'),

        sim,
        gate,
        after_gate,
    ])
