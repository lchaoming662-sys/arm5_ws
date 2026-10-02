#!/usr/bin/env python3
"""grasp_pipeline_mtc.py — O5-2 的抓放流水线（MoveIt Task Constructor）

把「生成抓取候选 → IK 过滤 → 接近 → 夹取 → 抬起 → 下压 → 张开 → 摘手 → 撤离」
串成一条流水线。这是 ROS 2 生态里 moveit_grasps 的对应物：moveit_grasps 只有
ROS 1 版本（仓库无 ros2 分支，apt 三个发行版都没有），MTC 才是官方在 ROS 2
上的那条路。

阶段链：
    CurrentState          取当前位形作为起点
    Connect               规划到预抓取位形
    GenerateGraspPose     绕物体批量生成候选抓取位姿（angle_delta 控制疏密）
    SimpleGrasp           对每个候选做 IK，并编排夹爪开合
    Pick                  接近 / 夹取 / 抬起
    Place                 下压 / 放置位姿 IK / 张开 / 摘手 / 撤离

前置：仿真在跑、move_group 在跑。省事的一条命令：
    ros2 launch arm_bringup bringup.launch.py

启动（**必须走 launch**）：
    ros2 launch arm_bringup grasp_mtc.launch.py
    ros2 launch arm_bringup grasp_mtc.launch.py display:=true
    ros2 launch arm_bringup grasp_mtc.launch.py execute:=true
    ros2 launch arm_bringup grasp_mtc.launch.py place:=false   只抓不起放（调试用）

为什么必须走 launch、不能直接 python 跑：
    MTC 要的是整套 MoveIt 参数 —— robot_description / robot_description_semantic
    / robot_description_kinematics / joint_limits / planning_pipelines，这些只能
    由 launch 注入。MTC 的 Python 绑定建在 rclcpp 上，而 rclcpp.Node 的 Python
    接口只有一个 name 属性，没有任何声明参数的途径，脚本自己配不出来。
    少任何一项都会以很隐蔽的方式退化（都是实测到的）：
      · 缺 kinematics.yaml   -> "No kinematics solver instantiated for group 'arm'"
                               → 25 个候选全部 IK 失败，看着像"这臂够不着"
      · 缺 ompl_planning.yaml -> 悄悄退化成 CHOMP 规划器，不报错，
                               但规划的已经不是我们配的那套
"""
import math
import os
import subprocess
import sys
import time

import rclcpp
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit.task_constructor import core, stages
from rclpy.action import ActionClient

_WS = os.path.expanduser('~/arm5_ws')

ARM_GROUP = 'arm'          # SRDF 里的规划组
EEF = 'gripper'            # SRDF 里的 <end_effector>
IK_FRAME = 'tcp'           # 两指夹持面中点，即真正的抓取点
OBJECT_ID = 'target_cube'  # 世界文件里的方块名，也是它在场景里的 id
ATTACH_LINK = 'claw_base'  # 方块抓起来后挂到哪个连杆上

# 执行用的控制器。名字必须与 arm_moveit_config/config/moveit_controllers.yaml
# 和 arm_bringup/config/arm_controllers.yaml 两边一致。
ARM_JOINTS = ('top_plate_joint', 'lower_arm_joint', 'upper_arm_joint',
              'wrist_joint', 'claw_base_joint')
GRIPPER_JOINT = 'right_claw_joint'
ARM_ACTION = '/arm_controller/follow_joint_trajectory'
GRIPPER_ACTION = '/gripper_controller/follow_joint_trajectory'
GRIPPER_GROUP = 'gripper'      # SRDF 里的夹爪组，MoveTo 开合都用它
OBJECT_XYZ = (0.0267, 0.0245, 0.0125)   # 世界文件里的真值（贴地 25 mm 立方体）

# 放回目标：**方块中心**在 base_link 下的目标 XY。
# 默认与 OBJECT_XYZ 相同，也就是「原位放回」。这不是偷懒 —— 旧演示
# tools/pick_demo_moveit.py 第 8 步也是原位放回（沿 Z 压回台面），保持一致
# 才能拿两边的抬起量/落点直接对比。
# 高度不在这里给：方块中心的高度恒为「贴台面」= OBJECT_XYZ[2]，
# 由 GRASP_TCP_Z 换算成 tcp 高度（见放回那一段）。
PLACE_XYZ = OBJECT_XYZ

# 抬起与下压共用一个距离。绑成一对：下压要正好把方块送回台面，
# 压过头会撞台面（伺服顶住不动，JTC 报 GOAL_TOLERANCE_VIOLATED）。
LIFT_DISTANCE = 0.050

# 速度缩放。关节段 0.40、笛卡尔段 0.25，两档都取自 tools/pick_demo_moveit.py
# 里已经实测标定过的那一组常数（那里记着实测依据）：
#   gz_ros2_control 的位置接口背后是软位置伺服，跟动参考时有与速度成正比的
#   滞后，实测 峰值跟踪误差 ≈ 0.20 × 缩放因子；而 arm_controller 的
#   trajectory 容差是 0.12 rad。0.40 → 0.08 rad，留 50% 余量。
#   笛卡尔段更吃精度，单独降到 0.25。
JOINT_VEL_SCALE = 0.40
JOINT_ACC_SCALE = 0.40
CART_VEL_SCALE = 0.25
CART_ACC_SCALE = 0.25
CART_MAX_STEP = 0.004      # 笛卡尔路径步长 4 mm

# 抓取/放置位姿下的 tcp 高度（相对台面），以及它相对【物体中心】的偏移。
# 物理含义是「在保证指尖不碰台面的前提下把抓手放到最低」，让指盒尽量多地
# 覆盖方块：
#     tcp 高度 = 台面 + 指尖间隙 + 指盒半长 = 0.0046 + 0.0185 = 0.0231 m
# 抓取时 GenerateGraspPose 生成的位姿原点在**物体中心**，而 tcp 是两指夹持面
# 中点，所以要补一个偏移；偏移取大了指尖会戳进台面 5.4 mm，实测报的是容差而
# 不是碰撞（见 docs/踩坑记录.md 第 34 条）。
FINGER_HALF_LEN = 0.0185
FINGER_TIP_CLEARANCE = 0.0046
GRASP_TCP_Z = FINGER_TIP_CLEARANCE + FINGER_HALF_LEN
GRASP_TCP_Z_OFFSET = GRASP_TCP_Z - OBJECT_XYZ[2]

# 规划失败时要汇报的阶段名，顺序与流水线一致（含 SimpleGrasp 容器内部的）。
# SimpleGrasp 是个 SerialContainer，MTC 的 Python 绑定没暴露 children()，
# 没法遍历，只能按名字查。
STAGE_NAMES = (
    'current', 'connect', 'approach object', 'grasp', 'generate grasp pose',
    'compute ik', 'close gripper', 'allow object collision', 'attach object',
    'lift object', 'move to place', 'lower object', 'open gripper',
    'forbid object collision', 'detach object', 'retreat after place',
)


def twist(frame, z):
    t = TwistStamped()
    t.header.frame_id = frame
    t.twist.linear.z = z
    return t


def to_sec(duration):
    """把 rclpy 的 Duration 换成秒。"""
    return duration.sec + duration.nanosec * 1e-9


def split_by_controller(subs):
    """把 MTC 解里的子轨迹按目标控制器分组，保持原有顺序。

    为什么必须自己分，而不是把整条解交给 move_group（Task.execute）：
      MTC 解里每条子轨迹的 TrajectoryExecutionInfo.controller_names **全是空的**。
      MTC 的 Python 绑定写不进这个属性 —— Property.setValue 对
      moveit_task_constructor_msgs/TrajectoryExecutionInfo 没有注册类型转换器，
      直接报 "No Python -> C++ conversion available"。
      一旦 controller_names 为空，move_group 只能把所有子轨迹合并成一条发给
      **默认控制器**（moveit_controllers.yaml 里 default: true 的 arm_controller）。

    这个失败方式极难从现象反推：Task.execute 返回 SUCCESS、机械臂也确实完整走完，
    唯一不对的是**夹爪一步没动**（/joint_states 里 right_claw_joint 始终是 0），
    于是方块纹丝不动。所以这里按关节名显式分发。
    """
    parts = []
    for sub in subs:
        jt = sub.trajectory.joint_trajectory
        if not jt.points:
            continue      # 纯改场景的子轨迹（allow / attach）没有轨迹可发
        names = list(jt.joint_names)
        if GRIPPER_JOINT in names:
            action = GRIPPER_ACTION
        elif names and all(n in ARM_JOINTS for n in names):
            action = ARM_ACTION
        else:
            print(f'  跳过无法分发的子轨迹：{names}', file=sys.stderr)
            continue
        parts.append((action, jt))
    return parts


def _run_goal(node, client, goal, timeout):
    """发一个 action goal 并等它结束。返回 True 表示 SUCCEEDED。"""
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
    handle = fut.result() if fut.done() else None
    if handle is None:
        print('    发 goal 超时', file=sys.stderr)
        return False
    if not handle.accepted:
        print('    goal 被拒绝', file=sys.stderr)
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut, timeout_sec=timeout)
    if not res_fut.done():
        print('    等结果超时', file=sys.stderr)
        return False
    status = res_fut.result().status
    if status != GoalStatus.STATUS_SUCCEEDED:
        print(f'    action 结束状态 {status}（{GoalStatus.STATUS_SUCCEEDED} = 成功）',
              file=sys.stderr)
        return False
    return True


def execute_solution(parts, timeout=180.0):
    """按顺序把每段轨迹发给它自己的控制器，逐段等执行完。

    这里用 rclpy 直接发 action，而不是走 move_group：控制器是**明确指定**的，
    不依赖 move_group 去猜。实测 rclpy 的节点可以和 MTC 的 rclcpp 节点在同一个
    进程里共存（两个 action 都可达、话题也能看到），所以不必为此拆出第二个进程。

    timeout 按墙上时间给：仿真 RTF 只有 ~0.48，一段 1.4 s 的轨迹实际要跑 3 s 上下，
    再留上"到点后等 goal 容差收敛"的时间。
    """
    rclpy.init()
    node = rclpy.create_node('mtc_solution_executor')
    try:
        arm_cli = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
        grip_cli = ActionClient(node, FollowJointTrajectory, GRIPPER_ACTION)

        for i, (action, jt) in enumerate(parts):
            is_gripper = action == GRIPPER_ACTION
            client = grip_cli if is_gripper else arm_cli
            if not client.wait_for_server(timeout_sec=10.0):
                print(f'  [{i + 1}] {action} 不可用', file=sys.stderr)
                return False

            # 夹爪也是 JointTrajectoryController（见 arm_controllers.yaml），
            # 所以两段动作用的是同一套 goal，不必把轨迹压成一个点。
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt

            if not _run_goal(node, client, goal, timeout):
                print(f'  [{i + 1}] 这一段失败，中止', file=sys.stderr)
                return False
        return True
    finally:
        node.destroy_node()
        rclpy.shutdown()


def report_failures(task, names, limit=3):
    """按阶段名打印失败原因。规划失败时调用。

    为什么需要这个：MTC 自己只报「最靠后的那个阶段失败」，
    前面真正的根因阶段被吞掉了。实测踩过两次：
      · 放回段用 GeneratePlacePose + ComputeIK，报的是
        「o5_2 pick and place: end interface (?) of 'place object'」，
        真正的错是上一行「place pose IK: interface of 'generate place
        pose' (← →) does not match external one (→ →)」—— 接口方向不匹配，
        和「物体够不着」毫无关系。
      · 候选抓取位姿全被 IK 否掉时，只报「没有可用解」，看不出是哪一环。

    实现说明：MTC 的 Python 绑定没有暴露 children()，没法遍历容器，
    所以只能按名字查（task['阶段名']）。查不到的阶段直接跳过。
    失败信息在 stage.failures 里，每个元素带 comment。
    """
    for name in names:
        try:
            stage = task[name]
        except Exception:       # noqa: BLE001 —— 名字不存在/不在本任务里，跳过
            continue
        fails = list(getattr(stage, 'failures', []) or [])
        if not fails:
            continue
        print(f'  [{name}] {len(fails)} 条失败：', file=sys.stderr)
        for f in fails[:limit]:
            comment = getattr(f, 'comment', '') or '(无说明)'
            print(f'      - {comment}', file=sys.stderr)
        if len(fails) > limit:
            print(f'      ... 另有 {len(fails) - limit} 条', file=sys.stderr)


def add_object_to_scene():
    """把目标加进 move_group 的规划场景。

    必须在规划之前、任务之外做：GenerateGraspPose 是从 scene() 查物体的，
    而 scene() 给的是**初始场景**，任务内部的 ModifyPlanningScene 只改向下传播
    的状态、到不了那里。实测报错就是 "object 'target_cube' not in scene"。
    官方 demo 用 moveit_commander 在外面加物体，也是同一个道理。
    """
    cmd = ('set +u; source /opt/ros/humble/setup.bash; '
           f'source {_WS}/install/setup.bash; '
           f'/usr/bin/python3 {_WS}/tools/scene_object.py --add')
    r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    return r.returncode == 0


def main():
    if len(sys.argv) < 2:
        print('本脚本需要 launch 注入 MoveIt 参数，请这样启动：\n'
              '  ros2 launch arm_bringup grasp_mtc.launch.py', file=sys.stderr)
        return 2

    rclcpp.init()
    # NodeOptions 这一句是整条流水线能不能执行的关键，别删。
    #
    # launch 传进来的参数是【覆盖项】(overrides)。rclcpp 默认
    # automatically_declare_parameters_from_overrides = False，此时
    # node.has_parameter("ompl.request_adapters") 返回 False —— 覆盖项不算
    # 「已声明」。MTC 建 PlanningPipeline 时正是用 has_parameter 去查适配器列表，
    # 查不到就静默地不加载任何请求适配器。
    #
    # 后果极其隐蔽：MoveIt 的时间参数化是由请求适配器
    # (default_planner_request_adapters/AddTimeOptimalParameterization) 完成的，
    # 适配器链一旦为空，规划出来的轨迹【每个点的时间戳都是 0】。
    # 规划阶段完全正常（有解、代价合理、RViz 里看着也对），只在执行时炸：
    #   [arm_controller] Time between points 0 and 1 is not strictly increasing,
    #                    it is 0.000000 and 0.000000 respectively
    #   [move_group] Goal request rejected -> 执行失败
    # 而 move_group 自己是 C++ 节点、用了会自动声明覆盖项的 NodeOptions，
    # 所以同一份参数在它那里一切正常 —— 对照之下很容易误判成"MTC 的问题"。
    #
    # 判据（都在日志里，一眼可辨）：
    #   正常：moveit.ros_planning.planning_pipeline 打印
    #         "Using planning request adapter 'Add Time Optimal Parameterization'"
    #   异常：只有 "Using planning interface 'OMPL'"，一条 adapter 都不打印
    opts = rclcpp.NodeOptions()
    opts.automatically_declare_parameters_from_overrides = True
    node = rclcpp.Node('grasp_pipeline_mtc', opts)

    if not add_object_to_scene():
        print('往场景里加物体失败，规划不可能成功。', file=sys.stderr)
        return 3

    # 【诊断分支】只生成候选并把它们打印出来，不做 IK。
    # 用途：确认 GenerateGraspPose 给的 25 个候选究竟是「从侧面抓」还是
    # 「从上面抓」。这条臂 5 自由度、只能俯抓，如果是侧向候选，
    # 后面 IK 全灭就有了确定解释。用 launch 的 probe:=true 打开。
    if os.environ.get('MTC_PROBE', '').lower() in ('1', 'true'):
        ox, oy, oz = OBJECT_XYZ
        probe = core.Task()
        probe.name = 'probe grasp poses'
        probe.loadRobotModel(node)
        probe.add(stages.CurrentState('current'))
        # GenerateGraspPose 的起始接口是「反向」(←)，不能直接接在 CurrentState
        # 的「正向」(→) 后面，否则报 cannot connect end interface。
        # Connect 是双向的，负责把两者接起来（主流水线里也有它）。
        probe_pipeline = core.PipelinePlanner(node)
        probe_pipeline.planner = 'RRTConnect'
        probe.add(stages.Connect('probe connect', [(ARM_GROUP, probe_pipeline)]))
        pg = stages.GenerateGraspPose('generate grasp pose')
        pg.object = OBJECT_ID
        pg.eef = EEF
        pg.angle_delta = math.pi / 12.0
        pg.pregrasp = 'open'
        pg.grasp = 'close'
        pg.setMonitoredStage(probe['current'])
        probe.add(pg)

        print('探查模式：只生成抓取候选并打印位姿 ...')
        if not probe.plan():
            print('连候选生成都失败了 —— 那说明问题在物体/场景，不在 IK。',
                  file=sys.stderr)
            return 1
        sols = probe['generate grasp pose'].solutions
        print(f'候选数：{len(sols)}')
        print(f'物体中心 (base_link) = ({ox:+.4f}, {oy:+.4f}, {oz:+.4f})')
        print('  序号  候选位置(x,y,z)                  相对物体中心(mm)          '
              '末端朝向 quat(x,y,z,w)')
        for i in range(len(sols)):
            try:
                tp = sols[i].start.properties['target_pose']
            except Exception as exc:      # noqa: BLE001 —— 取不到就说明 API 用法不对
                print(f'  #{i:2d}  取不到 target_pose：{exc}')
                continue
            p, q = tp.pose.position, tp.pose.orientation
            print(f'  #{i:2d}  ({p.x:+.4f},{p.y:+.4f},{p.z:+.4f})   '
                  f'({(p.x-ox)*1000:+7.1f},{(p.y-oy)*1000:+7.1f},'
                  f'{(p.z-oz)*1000:+7.1f})   '
                  f'({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})')
        time.sleep(0.5)
        return 0

    # launch 用环境变量把这个开关传进来（值形如 "true" / "false"）。
    # 关掉它就退回「只抓不起」的老行为 —— 放回这一段是新加的，
    # 出问题时能单独验证前半段，不用整体回退。
    do_place = os.environ.get('MTC_PLACE', '').lower() in ('1', 'true')

    task = core.Task()
    task.name = 'o5_2 pick and place' if do_place else 'o5_2 pick'
    task.loadRobotModel(node)   # 参数由 launch 注入，见文件头说明

    # ① 起点：当前位形
    task.add(stages.CurrentState('current'))

    # ② 规划到预抓取位形。规划器名取自 ompl_planning.yaml 里 arm 组的
    #    planner_configs 列表，就写 'RRTConnect'。
    #
    #    【更正】这里原先写的是 'RRTConnectkConfigDefault'，注释里断言
    #    「写成 RRTConnect 会 Cannot find planning configuration ... Will use
    #    defaults instead」—— 结论正好反了。实测（本轮，MoveIt 2 humble）：
    #    写 RRTConnectkConfigDefault 才会触发那句告警；写 RRTConnect 时
    #    告警消失，改打印
    #        Planner configuration 'arm[RRTConnect]' will use planner
    #        'geometric::RRTConnect'
    #    而且最优解代价从 6.53 掉到 3.89。
    #    也就是说这个名字从来没生效过，ompl_planning.yaml 里那套
    #    planner_configs 一直被忽略、规划器在用 OMPL 的裸默认参数 ——
    #    又一处「安静地给错答案」：规划照常成功，只是解变差。
    #    判据就一句日志：出现 "Will use defaults instead" 就是名字写错了。
    pipeline = core.PipelinePlanner(node)
    pipeline.planner = 'RRTConnect'
    # 速度缩放必须显式给，默认 1.0 会直接把执行打挂。
    # 实测的一条真实报文（默认值下）：
    #   [tolerances] State tolerances failed for joint 3:
    #                Position Error: -0.123133, Position Tolerance: 0.120000
    #   [arm_controller] Aborted due to state tolerance violation
    # 依据见文件顶部的 JOINT_VEL_SCALE / CART_VEL_SCALE 注释。
    # （这是被控对象的固有特性，不是故障；容差本身也别为了"跑通"而放宽，
    #   0.12 rad 是撞上东西时的故障网。）
    pipeline.max_velocity_scaling_factor = JOINT_VEL_SCALE
    pipeline.max_acceleration_scaling_factor = JOINT_ACC_SCALE
    connect = stages.Connect('connect', [(ARM_GROUP, pipeline)])
    task.add(connect)

    # ③ 生成候选抓取位姿。pregrasp / grasp 用 SRDF 里的 group_state 名，
    #    本工程夹爪语义是「0.0 张开、负方向闭合」。
    gen = stages.GenerateGraspPose('generate grasp pose')
    gen.object = OBJECT_ID
    gen.eef = EEF
    gen.angle_delta = math.pi / 12.0
    gen.pregrasp = 'open'
    gen.grasp = 'close'
    gen.setMonitoredStage(task['current'])

    # ④ 每个候选做 IK，并编排夹爪开合。
    #
    # ik_frame 的语义（见 MTC 的 setIKFrame 文档）：末端上用于 IK 的坐标系。
    # 这里 orientation.x = 1.0 是绕 X 转 180°，对应官方示例那句
    # `# grasp from top` —— **这一项不是可选项**：
    # 用 probe 模式实测过 GenerateGraspPose 的 25 个候选，位置都是 (0,0,0)
    # （物体参考系原点 = 物体中心），朝向是绕**物体 Z 轴**均匀转一周
    # （15° 一步）。绕 Z 转意味着候选的 Z 轴都指向世界 +Z（朝上），
    # 而 tcp 的 +Z 是「手指伸出方向」—— 手指朝上就绝对抓不到地面上的东西。
    # 实测：不翻这个方向 → 25 个候选全部 IK 失败，而同一位置直接调
    # /compute_ik 服务却是成功的，很容易误判成"这臂够不着"。
    ik = PoseStamped()
    ik.header.frame_id = IK_FRAME
    ik.pose.orientation.x = 1.0
    ik.pose.orientation.w = 0.0
    # 还有一个容易漏掉的东西：**抓取点不在方块中心**。
    #
    # GenerateGraspPose 生成的位姿原点就在【物体中心】（probe 模式实测：25 个
    # 候选位置全是 (0,0,0)，即物体参考系原点）。可是 tcp 是两指夹持面中点，
    # 指盒沿指长方向有 ±18.5 mm。把 tcp 直接放到方块中心（离台面 12.5 mm）
    # 会让指尖伸到台面以下 5.4 mm —— 物理上顶住桌面，伺服永远收敛不了，执行
    # 时报的是容差而不是碰撞：
    #     [tolerances] State tolerances failed for joint 2:
    #                  Position Error: 0.027348, Position Tolerance: 0.020000
    #     [arm_controller] Aborted due to goal_time_tolerance exceeding by 1.008657 seconds
    # 注意这个偏差与速度**无关**：把笛卡尔段缩放从 1.0 降到 0.25，误差
    # 0.0269 → 0.0273 几乎不变。降速治不了它，别在这上面绕。
    # 判据（用 tools/grasp_geometry.py 的 FK 一算就出来）：
    #     tcp = 方块中心 (0.0267, 0.0245, 0.0125)，
    #     right_claw 碰撞盒 z ∈ [-0.0054, +0.0304] —— 负值就是戳进台面。
    #
    # 正确高度直接沿用 tools/pick_demo_moveit.py 已经标定过的常数：
    #     抓取位 TCP 高度 = 台面 + 指尖间隙 + 指盒半长 = 0.0046 + 0.0185 = 0.0231 m
    # 物理含义是「在保证指尖不碰台面的前提下把抓手放到最低」，让指盒尽量多地
    # 覆盖方块。也就是比方块中心高 0.0231 - 0.0125 = 0.0106 m。
    ik.pose.position.z = GRASP_TCP_Z_OFFSET

    grasp = stages.SimpleGrasp(gen, 'grasp')
    grasp.setIKFrame(ik)

    # ⑤ 接近 / 夹取 / 抬起。两段都走 base_link 的竖向：接近向下、抬起向上。
    #
    # 两个距离参数是 (min, max)：min 是「至少要能走的距离」，达不到就判失败；
    # max 是「最多退开多少」。这里 min 特意取得小 —— 这条臂的抓取位形本身
    # 就贴着关节限位（实测抓取时 wrist = -1.409、claw_base = 0.796，限位是 ±1.50），
    # 要求退开 24 mm 时笛卡尔路径只走得出 8 mm，就被 min_fraction 判失败了。
    # 拿实际可达空间说话，而不是照抄 panda 的 0.03。
    # ⑤ 接近 —— 笛卡尔运动，必须单独降速（见文件顶部 CART_VEL_SCALE 的注释）。
    #
    # 这里不用 MTC 的 Pick 容器，而是把它的四个动作摊开自己搭。原因：
    #   Pick 内部自建的 CartesianPath 求解器**写死用默认速度缩放 1.0**，
    #   而 Python 绑定拿不到那个求解器对象（它是 C++ 成员、构造函数里 create
    #   出来的），ContainerBase 的 remove() 又因为 unique_ptr 归属冲突直接抛
    #   "Invalid unique_ptr: another instance owns this pointer already"。
    #   摊开之后求解器由我们创建，缩放是我们定 —— 顺带阶段链也更清楚。
    # Pick 原本的动作顺序（实测子轨迹顺序 + 失败阶段名反推）：
    #   接近 → ComputeIK → 闭爪 → 允许碰撞 → 附着 → 抬起
    cartesian = core.CartesianPath()
    cartesian.max_velocity_scaling_factor = CART_VEL_SCALE
    cartesian.max_acceleration_scaling_factor = CART_ACC_SCALE
    cartesian.step_size = CART_MAX_STEP

    approach = stages.MoveRelative('approach object', cartesian)
    approach.group = ARM_GROUP
    approach.setDirection(twist('base_link', -1.0))   # 沿 base_link 的 -Z 下压
    # (min, max) 距离。min 特意取得小：这条臂的抓取位形本身就贴着关节限位
    # （实测抓取时 wrist = -1.409、claw_base = 0.796，限位 ±1.50），要求退开
    # 24 mm 时笛卡尔路径只走得出 8 mm，会被 min_fraction 判失败。
    # 拿实际可达空间说话，而不是照抄 panda 的 0.03。
    approach.min_distance = 0.001
    approach.max_distance = 0.050
    task.add(approach)

    # ⑥ IK + 闭爪。SimpleGrasp 是个容器，内部就是 ComputeIK 加一段 MoveTo。
    #    eef / object 原本由 Pick 通过 PARENT 传给子阶段；单独用就得自己设，
    #    不设会直接规划失败（ComputeIK 拿不到末端执行器）。
    grasp.eef = EEF
    grasp.object = OBJECT_ID
    task.add(grasp)

    # ⑦ 允许手指与方块接触，再把方块挂到手上。
    #    手指是压进方块的，不显式豁免，attach 会被判成碰撞：
    #    「attach object: target_cube colliding with left_claw」。
    allow = stages.ModifyPlanningScene('allow object collision')
    allow.allowCollisions(OBJECT_ID, True)
    task.add(allow)

    attach = stages.ModifyPlanningScene('attach object')
    attach.attachObject(OBJECT_ID, ATTACH_LINK)
    task.add(attach)

    # ⑧ 抬起：同样是笛卡尔运动，同样走降速后的求解器。
    lift = stages.MoveRelative('lift object', cartesian)
    lift.group = ARM_GROUP
    lift.setDirection(twist('base_link', 1.0))
    lift.min_distance = 0.001
    lift.max_distance = LIFT_DISTANCE
    task.add(lift)

    # ---------------------------------------------------------------------
    # ⑨ 放回（Place）
    #
    # 拓扑与官方 demo 的 place 段一致：
    #     移到放置点上方 → 下压 → 张开 → 恢复碰撞检查 → 摘手 → 撤离
    # 顺序有硬依赖，不能调换：
    #   · 下压必须在 detach 之前，否则方块在半空就掉下来；
    #   · 张开必须在 forbid 之前 —— 手指还压在方块上，先恢复碰撞检查会
    #     让「张开」这一段被判成碰撞；
    #   · detach 必须在张开之后，否则夹爪张开了方块还挂在手上。
    #
    # 与官方的一处**有意**不同：官方用 GeneratePlacePose + ComputeIK 来指定
    # 放置位姿（pose 属性给的是物体中心的位置），本工程改成用 MoveTo 直接给
    # tcp 的目标位姿。原因是接口方向：
    #   GeneratePlacePose 是反向(←)接口的生成器，ComputeIK 是 WrapperBase，
    #   而 WrapperBase 的接口检查是【严格相等】的（ParallelContainerBase::
    #   validateInterfaces）。把它放进串行容器的非首位，父级期望的接口与子阶段
    #   的接口对不上，实测直接报：
    #     place pose IK: interface of 'generate place pose' (← →) does not
    #     match external one (→ →).
    #   改成 MoveTo(PoseStamped) 之后整条 place 链全是正向(→ →)阶段，可以像
    #   pick 链那样直接挂在 task 上，且每一段都产生真实轨迹 —— 官方那套里
    #   ComputeIK 只改状态、不产生轨迹，机械臂并不会真的动到 IK 解。
    #
    # 放置点用 tcp 位姿表达而不是方块中心：tcp 落在「台面 + 指尖间隙 + 指盒
    # 半长」= GRASP_TCP_Z 时，方块正好贴台面（与抓起时同一几何关系）。
    # ---------------------------------------------------------------------
    if do_place:
        place_tcp = PoseStamped()
        place_tcp.header.frame_id = 'base_link'
        place_tcp.pose.position.x = PLACE_XYZ[0]
        place_tcp.pose.position.y = PLACE_XYZ[1]
        # 先抬到放置点上方 LIFT_DISTANCE 处再下压：关节空间搬运时方块离台面
        # 足够高，不会一路刮着桌面过去。默认 PLACE_XYZ 与抓取点相同，此时
        # 这一段就是纯竖直上移，路径必然无碰撞。
        place_tcp.pose.position.z = GRASP_TCP_Z + LIFT_DISTANCE
        place_tcp.pose.orientation.w = 1.0

        # ⑨-1 把 tcp 搬到放置点上方。**必须用笛卡尔求解器**：
        #     MoveTo 收到 PoseStamped 目标时走的是笛卡尔分支，调的是
        #       planner_->plan(scene, link, offset, target, jmg, ...)
        #     这个重载只有 CartesianPath 实现得了。喂 OMPL 的 PipelinePlanner
        #     进去不会报「接口不对」，只会一路走到 OMPL 然后
        #       [ompl] arm/arm: Unable to sample any valid states for goal tree
        #       Invalid goal state
        #       Failing stage(s): move to place (0/13): GOAL_STATE_INVALID
        #     —— 13 条输入全灭，看起来像「目标不可达」，其实是调用方式错了。
        #     别用 /compute_ik 去验这个目标：笛卡尔分支压根不做 IK。
        #     默认 PLACE_XYZ 与抓取点相同时，这一段就是竖直上移 LIFT_DISTANCE，
        #     与抬起的路径重合，稳。
        transport = stages.MoveTo('move to place', cartesian)
        transport.group = ARM_GROUP
        transport.setGoal(place_tcp)
        task.add(transport)

        # ⑨-2 下压：笛卡尔直线下降，方块落到台面。
        #     max_distance 与抬起的 LIFT_DISTANCE 配对。
        #     min 依然取得小，理由同 approach：这条臂的位形贴着限位，
        #     要求走满 50 mm 时笛卡尔路径可能只走得出 8 mm。
        lower = stages.MoveRelative('lower object', cartesian)
        lower.group = ARM_GROUP
        lower.setDirection(twist('base_link', -1.0))
        lower.min_distance = 0.001
        lower.max_distance = LIFT_DISTANCE
        task.add(lower)

        # ⑨-3 张开夹爪。用 JointInterpolationPlanner 而不是 OMPL：
        #     夹爪组只有 1 个自由度（left_claw_joint 是 mimic，不进组），
        #     插值就是精确解，走 OMPL 反而是无谓的搜索。
        #     goal='open' 是 SRDF 里的 group_state 名。
        open_gripper = stages.MoveTo('open gripper', core.JointInterpolationPlanner())
        open_gripper.group = GRIPPER_GROUP
        open_gripper.setGoal('open')
        task.add(open_gripper)

        # ⑨-4 恢复手指与方块的碰撞检查。抓取时豁免过（手指是压进方块的），
        #     现在方块要离开手指了，得把豁免撤掉，否则撤离时会带着一个
        #     「永久可穿透」的物体 planning，碰撞检查等于形同虚设。
        forbid = stages.ModifyPlanningScene('forbid object collision')
        forbid.allowCollisions(OBJECT_ID, False)
        task.add(forbid)

        # ⑨-5 摘下方块：方块从「挂在爪上」变回「躺在台面上」。
        release = stages.ModifyPlanningScene('detach object')
        release.detachObject(OBJECT_ID, ATTACH_LINK)
        task.add(release)

        # ⑨-6 撤离：手往上抬，把方块留在台面上。
        retreat = stages.MoveRelative('retreat after place', cartesian)
        retreat.group = ARM_GROUP
        retreat.setDirection(twist('base_link', 1.0))
        retreat.min_distance = 0.001
        retreat.max_distance = LIFT_DISTANCE
        task.add(retreat)

    print(f'目标 {OBJECT_ID}，5 自由度臂 + position_only_ik，开始规划 ...')
    t0 = time.time()
    ok = task.plan()
    print(f'规划耗时 {time.time() - t0:.1f} s，结果：{"成功" if ok else "失败"}')

    if not ok:
        print('没有可用解。下面是逐阶段的失败原因 —— 别只看最后一行，'
              'MTC 只会报「最靠后的那个阶段失败」，根因往往在更前面。',
              file=sys.stderr)
        report_failures(task, STAGE_NAMES)
        return 1

    solutions = task.solutions
    print(f'解的数量：{len(solutions)}')
    best = solutions[0]
    print(f'最优解代价：{best.cost}')

    # 解的构成：子轨迹 → 关节 → 时长。执行就按这个顺序逐段发。
    try:
        subs = list(best.toMsg().sub_trajectory)
    except Exception as exc:          # noqa: BLE001
        print(f'取解的子轨迹失败：{exc}', file=sys.stderr)
        return 5
    parts = split_by_controller(subs)
    print(f'子轨迹 {len(subs)} 条，其中 {len(parts)} 段带轨迹：')
    for i, (action, jt) in enumerate(parts):
        print(f'  [{i + 1}] {action}  '
              f'关节={list(jt.joint_names)}  点数={len(jt.points)}  '
              f'时长={to_sec(jt.points[-1].time_from_start):.2f} s')

    # launch 通过环境变量把这两个开关传进来（值形如 "true" / "false"）
    if os.environ.get('MTC_DISPLAY', '').lower() in ('1', 'true'):
        task.publish(best)
        print('已发到 RViz 的 MTC 面板（Motion Planning Tasks）')

    if os.environ.get('MTC_EXECUTE', '').lower() in ('1', 'true'):
        print('开始执行 ...')
        if not execute_solution(parts):
            print('执行失败', file=sys.stderr)
            return 4
        print('执行完成')

    time.sleep(0.5)   # 让 introspection 的消息发完再退出
    return 0


if __name__ == '__main__':
    sys.exit(main())
