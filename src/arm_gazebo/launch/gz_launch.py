#!/usr/bin/env python3
"""gz_launch.py — 在 Gazebo Fortress 里启动 O5-2 机械臂仿真

启动链（顺序很重要）：

    ① robot_state_publisher  把 xacro 展开成 URDF，发布 /robot_description 与 TF
    ② Gazebo Fortress        -r 表示「立即开始跑」，不要停在暂停状态
    ③ parameter_bridge       把 gz 的 clock / camera_info / points 桥到 ROS 2
    ④ image_bridge           把腕部相机的彩图与深度图桥到 ROS 2（camera:=false 时不起）
    ⑤ ros_gz_sim create      从 /robot_description 话题把机器人生成进世界
    ⑥ controller spawner     依次激活 joint_state_broadcaster 与两个轨迹控制器

关于 ⑥ 的顺序：controller_manager 由 gz_ros2_control 插件在仿真进程内注册，
所以外部**不能**再起 ros2_control_node（会撞名）。spawner 会等这个服务出现，
因此不必加固定延时——用 OnProcessExit 串联，spawn 真的完成了才走下一步。

==============================================================================
怎么让仿真跑得快一点
==============================================================================
本机是软件渲染（llvmpipe，没有 GPU），仿真的实时因子（RTF）是主要瓶颈。
三个开关，按收益从大到小：

    gui:=false      只起 server，不开界面。本机 Gazebo GUI 常驻占 ~1.7 个核，
                    关掉后 RTF 明显提升。批量验收、录数据时用这个。
                    （看不到窗口，但话题/服务/控制器全都在。）

    camera:=false   整个腕部相机不进模型。Gazebo 的 Sensors 系统在 PostUpdate
                    里等渲染线程出图，相机会拖住整个仿真循环。

    相机规格        默认已从 640x480@30Hz 降到 320x240@10Hz（见 arm_camera.xacro）。

另外 MoveIt 侧的速度由 arm_moveit_config/config/joint_limits.yaml 里的
max_velocity / max_acceleration 决定，与这里无关。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_desc = get_package_share_directory('arm_description')
    pkg_gz = get_package_share_directory('arm_gazebo')

    # URDF 里写的是 package://arm_description/meshes/*.stl。
    # Gazebo 不认 package://，它会把这类 URI 当成 model:// 去资源路径里找
    # 一个叫 arm_description 的目录。所以必须把 <install>/share 这一层
    # 告诉它，否则网格全部加载失败——表现为 gzclient 里只有一根根空骨架
    # （视觉丢失），而且更麻烦的是【碰撞体也一起丢了】，物理形同虚设。
    # 两个变量名都要设：Fortress 认 IGN_GAZEBO_RESOURCE_PATH，
    # 新一点的 gz 认 GZ_SIM_RESOURCE_PATH。
    share_parent = os.path.join(pkg_desc, os.pardir)
    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH', value=share_parent)
    set_resource_path_gz = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH', value=share_parent)

    xacro_file = os.path.join(pkg_desc, 'urdf', 'arm_gz.urdf.xacro')
    world_file = os.path.join(pkg_gz, 'worlds', 'arm_world.sdf')
    bridge_config = os.path.join(pkg_gz, 'config', 'arm_bridge.yaml')
    gz_sim_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')

    gui = LaunchConfiguration('gui')
    camera = LaunchConfiguration('camera')

    # ① 模型 → TF 树 + /robot_description 话题
    #    camera:=false 时整个相机 link 都不进模型，传感器也就不会被创建
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file, ' use_camera:=', camera]),
            value_type=str)
    }
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[robot_description, {'use_sim_time': True}],
               output='screen')

    # ② 启动仿真器。-r = 立即运行（不加会是暂停状态，时钟不走、/joint_states 无数据）
    #    gui:=false 时加 -s = 只跑 server（无界面）
    gz_args = ['-r -v4 ', LaunchConfiguration('world')]
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            'gz_args': gz_args,
            'on_exit_shutdown': 'true',
        }.items(),
        condition=IfCondition(gui),
    )
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            'gz_args': gz_args + [' -s'],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=UnlessCondition(gui),
    )

    # ③ ROS ←→ Gazebo 话题桥（clock / camera_info / 等）
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
                  output='screen')

    # ④ 相机图片桥（相机没开就别起了）
    #    rgbd_camera 出彩图与深度图两张，都走这里；点云与内参在 ③ 的 yaml 里。
    image_bridge = Node(package='ros_gz_image', executable='image_bridge',
                        arguments=['/wrist_cam/image', '/wrist_cam/depth_image'],
                        output='screen',
                        condition=IfCondition(camera))

    # ⑤ 把机器人生成进世界（从 /robot_description 话题读模型）
    spawn = Node(package='ros_gz_sim', executable='create',
                 arguments=['-topic', 'robot_description',
                            '-name', 'o5_2_arm',
                            '-x', '0.0', '-y', '0.0', '-z', '0.0'],
                 output='screen')

    # ⑥ 三个控制器
    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name,
                               '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '120',
                               '--service-call-timeout', '60'],
                    output='screen')

    jsb = spawner('joint_state_broadcaster')
    arm_ctrl = spawner('arm_controller')
    gripper_ctrl = spawner('gripper_controller')

    after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn,
            on_exit=[
                LogInfo(msg='机器人已生成，开始加载 joint_state_broadcaster'),
                jsb,
            ],
        )
    )

    after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=jsb,
            on_exit=[
                LogInfo(msg='joint_state_broadcaster 就绪，开始加载两个轨迹控制器'),
                arm_ctrl,
                gripper_ctrl,
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=world_file,
            description='要加载的 SDF 世界文件路径'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='是否打开 Gazebo 界面。false = 只跑 server，仿真更快'),
        DeclareLaunchArgument(
            'camera', default_value='true',
            description='是否加载腕部相机（含传感器与图片桥）。false 可显著提速'),
        # 必须排在最前面：环境变量只对「它之后启动的进程」生效
        set_resource_path,
        set_resource_path_gz,
        rsp,
        gazebo,
        gazebo_headless,
        bridge,
        image_bridge,
        spawn,
        after_spawn,
        after_jsb,
    ])
