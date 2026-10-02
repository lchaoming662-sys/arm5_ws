#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pick_demo_moveit.py — 用 MoveIt 2 规划驱动的抓取闭环演示

用法（三个终端，各自先 source /opt/ros/humble/setup.bash 与 ~/arm5_ws/install/setup.bash）：

    ros2 launch arm_gazebo  gz_launch.py            # A：Gazebo 仿真（含控制器）
    ros2 launch arm_moveit_config demo.launch.py    # B：MoveIt move_group
    /usr/bin/python3 ~/arm5_ws/tools/pick_demo_moveit.py   # C：本脚本

可选参数：
    --dry-run        只规划不执行（验证规划可行性，不动仿真）
    --no-scene       不把方块登记进规划场景（对照实验用）
    --hold           抬起后停住不退场（截图用）
    --approach H     预抓取高度，默认 0.060 m
    --lift H         抬起高度，默认 0.130 m

===============================================================================
和 tools/pick_demo.py（旧版）的区别
===============================================================================
旧版把关节角**写死**在脚本里，直接丢给 arm_controller：

    写死关节角 → 关节空间插值 → 圆弧下压 → 抓

本版让 MoveIt 真正参与决策：

  ① 目标位姿由**方块在 Gazebo 里的实测位姿**推出来，不是写死的常数。
     方块挪个地方，旧版要改代码，本版不用。

  ② 下压 / 抬起走 **/compute_cartesian_path 的竖直直线**，而不是关节空间圆弧。
     旧版的"下压"其实带着 12.6 mm 的横向偏移（关节插值跑出来的弧线）；
     抓取要的是竖直进刀，横向偏移会让手指蹭到方块侧面。
     旧版抬起位还把夹爪倾转了 60.8°，本版抬起全程夹爪朝下。

  ③ 路径由 **OMPL 规划**，方块登记进规划场景作为障碍物，关节限位与速度缩放
     由 MoveIt 统一把关。

  ④ 位姿目标走 **position-only IK**。本臂只有 5 个自由度，KDL 默认做完整
     6D 位姿 IK 是数学上无解的（见 config/kinematics.yaml）。

有一处**故意不用** MoveIt：夹爪开合。单自由度的 gripper 组规划不出任何东西，
而 GripperCommand 会回报 `stalled` / `reached_goal`——那正是抓取需要的
"夹到东西了"接触反馈。这是取舍，不是遗漏。
（另外 `left_claw_joint` 是 URDF 里的 mimic 从动指，本来也不接受指令。）

===============================================================================
验收判据（不看 ROS 层关节值，直接问 Gazebo）
===============================================================================
  · 笛卡尔路径 fraction 必须 = 1.0（否则说明直线中间有段规划不出来）
  · 抬起后 `ign model -m target_cube -p` 的 z 必须明显上升
  · 放回后方块与初始位置误差应在毫米级
"夹爪闭合了但方块滑掉了"不会被漏掉——判定读的是方块，不是关节。
"""

import argparse
import math
import re
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from geometry_msgs.msg import Point, Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             PlanningOptions, PlanningScene, PositionConstraint,
                             RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionFK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

# =============================================================================
# 常量：全部由 tools/grasp_geometry.py 的 FK 结果推导 / 校核
# =============================================================================

ARM_JOINTS = ['top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 'wrist_joint', 'claw_base_joint']
ALL_JOINTS = ARM_JOINTS + ['right_claw_joint', 'left_claw_joint']

FRAME = 'world'
EE_LINK = 'tcp'            # SRDF 里 arm 组的末端（不是 claw_base！见 o5_2_arm.srdf）
PLANNING_GROUP = 'arm'

# --- 夹爪几何（与 arm_core.xacro 里 right_claw / left_claw 的碰撞盒一致）-------
FINGER_HALF_LEN = 0.0185        # 指碰撞盒 0.012 x 0.012 x 0.037，沿指长轴半长
FINGER_TIP_CLEARANCE = 0.0046   # 抓取时指尖离台面的间隙
# 抓取位 TCP 高度 = 台面 + 指尖间隙 + 指盒半长。
# 物理含义：**在保证指尖不碰台面的前提下把抓手放到最低**，让指盒尽量多地
# 覆盖方块。FK 实测校核值 0.023068 m（与 0.0231 差 0.03 mm）。
GRASP_TCP_Z = FINGER_TIP_CLEARANCE + FINGER_HALF_LEN

GRIP_OPEN = 0.0        # 两指竖直张开位，夹持面间距 29.28 mm
# 闭合指令。这个数**必须按几何算**，不能拍。
#
# 在抓取位形下，「两指夹持面间距」随 jaw_a 角的变化（tools/grasp_geometry.py 算得）：
#
#     jaw_a 角      面间距      对方块(25 mm)的压入量
#      0.00        29.28 mm      -4.28 mm（还没碰到）
#     -0.1465      25.00 mm       0.00 mm  ← 刚好贴住
#     -0.20        23.29 mm      +1.71 mm
#     -0.35        17.77 mm      +6.87 mm
#
# 注意：原来这里写 -0.35，等于命令手指**多压进方块 6.9 mm**。方块是刚体、压不进去，
#    物理引擎只能把它往外挤 —— 实测夹爪一闭合方块就被推开 11 mm，
#    于是"夹爪闭上了但方块没被抓住"，而且手指会被判定 stalled。
#    把物理步长从 1 ms 放到 3 ms 后接触解算更硬，这个问题从"偶尔失败"变成"经常失败"。
#
#    取 -0.20：只命令约 1.7 mm 的余量，足够建立法向力（接触刚度 kp=1e6 N/m，
#    0.1 mm 就是 ~100 N 量级），又不会把方块挤跑。
GRIP_CLOSE = -0.20
GRIP_EFFORT = 20.0     # 工程取值，不是舵机真实能力

CUBE_ID = 'target_cube'
CUBE_SIZE = 0.025      # 与世界文件 arm_world.sdf 里的 box 一致
# 方块被抓到后挂到哪个连杆上，以及哪些连杆允许与它接触。
# 手指必须列进 touch_links —— 夹持面是压进方块的，不豁免就还是"碰撞"。
ATTACH_LINK = 'claw_base'
TOUCH_LINKS = ('claw_base', 'tcp', 'right_claw', 'left_claw')
# 世界文件里方块的初始位姿。演示开始前用它复位，保证从确定的初始状态出发
# （**重要**：跑过几轮演示后方块常被碰歪，不复位的话抓取高度就对不上了）。
CUBE_HOME = (0.0267, 0.0245, 0.0125)
# 开局清场时方块被临时扔到哪（在臂工作空间之外）
CUBE_PARK = (0.55, 0.55, 0.05)
WORLD_NAME = 'arm_world'

CUBE_LIFT_THRESHOLD = 0.05     # 抬升超过 5 cm 才算真被提起来
# 放回时，让方块**最低点**停在离地这么高，再张开夹爪、让它自己落下去。
# 为什么不能像抓起时那样直接命令 TCP 回到原抓取高度：
#   方块是刚体、被夹着抬起时还会带一点点倾角，命令 TCP 回到原高度等于
#   把它往地面里按 —— 伺服永远收敛不了，JTC 会以
#   "goal_time_tolerance exceeding" 收场（实测 TCP 差 2.9 mm 到不了目标）。
#   实测 8 mm 足够：方块自重 20 g，落下 8 mm 对最终落点影响在毫米级。
PLACE_CLEARANCE = 0.008

# --- 规划参数 --------------------------------------------------------------
PLAN_TIME = 8.0          # 单次规划时间上限（s）
PLAN_ATTEMPTS = 5
# 速度 / 加速度缩放。实际值 = joint_limits.yaml 里的 max_* × 这里的系数。
#
# 注意：0.40 是**实测**选出来的，不是拍的。gz_ros2_control 的位置接口背后是
#    一套软位置伺服，跟动参考时会留下与速度成正比的滞后（实测 ≈ 0.20 × 缩放）：
#
#        s=0.15 → 0.030 rad    s=0.40 → 0.080 rad  ← 选这档
#        s=0.20 → 0.040 rad    s=0.50 → 0.100 rad
#        s=0.30 → 0.060 rad    s=0.60 → 0.120 rad（顶到容差，被掐断）
#
#    arm_controllers.yaml 里的 trajectory 容差是 0.12 rad，
#    选 0.40 有 50% 余量；再往上（0.5）余量就只剩 20% 了，不值当。
#    实测同一段 home→预抓取：s=0.20 要 12.9 s，s=0.40 只要 6.9 s。
VEL_SCALE = 0.40
ACC_SCALE = 0.40
# 注意：**笛卡尔段要单独降速**。
#    同一条臂、同样的限位，/compute_cartesian_path 出的轨迹比关节空间规划
#    更吃跟踪精度：实测速度 0.40 时下压段会**间歇性**报
#    "path tolerance violation"（同一段有时过、有时不过，最难受的一种失败）。
#    原因在于笛卡尔路径先按 4 mm 密采样再时间参数化，关节速度曲线更陡。
#    下压/抬起都只有几厘米，降速的代价很小，换来的是稳定。
#    0.25 相对 0.40 留了 1.6 倍余量。
CART_VEL_SCALE = 0.25
CART_ACC_SCALE = 0.25
CART_MAX_STEP = 0.004    # 笛卡尔路径步长 4 mm
POS_TOL = 0.002          # 位姿目标的球半径 2 mm

# 「回零位 / 收复姿态」专用速度，比 VEL_SCALE 低一档。
# 与第 0 步的"恢复回零位"保持一致：收尾动作不在关键路径上，慢一点没有代价。
#
# 注意：降速**并不能**解决回零位失败。
#    实测过 0.40 与 0.20 两档，都在同一处被 JTC 掐断（见下），
#    所以不要以为"把速度再压低一点就好了"。真正的现象是：
#
#      · 报错点：arm_controller 的【state/path 容差】检查，落在 wrist_joint
#        （控制器关节表索引 3）
#      · 表现：wrist_joint 的实际位置**一动不动**（采样里 12 s 内位移 0.0000，
#        速度恒为 0），而参考值以约 0.1~0.24 rad/s 正常推进
#      · 于是误差线性累积，越过 0.12 rad 就被 ABORT，
#        MoveIt 只报 "Solution found but controller failed during execution" +
#        CONTROL_FAILED(-4)，报错点离根因很远
#
#    也就是说：这不是"跟不动"，是**那个姿态下腕关节物理上不动**。
#    用 tools/jtc_trace.py 采 /arm_controller/state 的 error 可以一眼看出来
#    （误差是**单调线性上涨**的，不是抖动）。
#
#    已经排除的原因（都实测过）：
#      排除 速度太快           —— 0.20 也一样失败
#      排除 腕关节坏了         —— 从零位走到 -1.5708 再回来，SUCCEEDED
#      排除 ABORT 后控制器死锁 —— 违规之后再做同样的慢速移动，SUCCEEDED
#      排除 姿态本身可复现     —— 拿着当时那组关节角重放，SUCCEEDED
#      成立 与"被顶到限位上"相关：失败时腕关节正好压在 URDF 下限
#        -1.5708 上（实测值 -1.5708000110）。这是**锁存状态**，
#        重启仿真才能恢复 —— 与踩坑记录第 26 条是同一类现象。
#
#    所以本脚本在回零位失败时给出的建议仍然是有效的：重启仿真。
#    下一步的真修法方向见 docs/踩坑记录.md 第 31 条（给腕关节留限位余量）。
HOME_VEL_SCALE = 0.20

# 执行类调用超时。仿真比真实时间慢 6~8 倍，必须给足
T_EXEC_MOVE = 420.0
T_EXEC_CART = 420.0
T_EXEC_GRIP = 240.0


def _sh(cmd, timeout=60):
    return subprocess.run(['bash', '-lc', cmd], capture_output=True,
                          text=True, timeout=timeout).stdout


def read_cube_pose_full():
    """直接问 Gazebo 方块的**真实位姿**，不经过 ROS。返回 (xyz, rpy)。

    `ign model -m <名字> -p` 输出两行：位置 与 RPY（弧度）。
    姿态也要读，是因为下压放回时要按它算方块最低点（见 place_target_z）。
    """
    out = _sh('timeout 25 ign model -m %s -p' % CUBE_ID, timeout=60)
    rows = re.findall(r'\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', out)
    if len(rows) < 2:
        print('!! 读不到方块位姿（仿真在跑吗？）：')
        print(out)
        sys.exit(1)
    xyz = [float(v) for v in rows[0]]
    rpy = [float(v) for v in rows[1]]
    return xyz, rpy


def read_cube_pose():
    """只要位置，保持旧签名。"""
    return read_cube_pose_full()[0]


def cube_support_half_z(rpy):
    """方块在**当前姿态**下，沿世界 Z 方向的支撑半长。

    立方体半边长 h，姿态旋转矩阵第三行 r，则它沿 ±Z 的最大伸出是
        h * (|r_zx| + |r_zy| + |r_zz|)
    这是支撑函数（support function）对长方体的取值。

    为什么需要它：方块被夹着抬起时会带一点点倾角。平放（rpy=0）时沿 Z 的
      支撑半长就是 h，倾斜后会**变大**（实测 0.0125 → 0.01495）。
      不知道这个量，"该降到多低"就无从算起。
    """
    r, p_, y = rpy
    # R = Rz(y)·Ry(p)·Rx(r) 的第三行
    row = [-math.sin(p_),
           math.cos(p_) * math.sin(r),
           math.cos(p_) * math.cos(r)]
    return CUBE_SIZE / 2.0 * sum(abs(v) for v in row)


def set_cube_pose(xyz):
    """把方块瞬移到位姿 xyz（Gazebo 的 set_pose 服务）。"""
    req = ('name: "%s" position: {x: %.5f, y: %.5f, z: %.5f} orientation: {w: 1.0}'
           % (CUBE_ID, xyz[0], xyz[1], xyz[2]))
    out = _sh('timeout 20 ign service -s /world/%s/set_pose '
              '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
              '--timeout 5000 --req \'%s\'' % (WORLD_NAME, req), timeout=40)
    if 'data: true' not in out:
        print('    注意：复位服务返回异常（不影响继续，但初始状态可能不干净）')
        print('    ', out.strip()[:200])


def reset_cube():
    """把方块放回世界文件里的初始位姿。

    跑过几轮之后方块常被手指碰歪、甚至漂到别处（实测漂了 19 mm 并带
    0.27 rad 翻滚角）。位姿由脚本自己读，理论上歪着也能抓，但**歪着的方块
    抓取高度是错的**（方块中心抬高、夹持面不再竖直），所以演示前要复位。
    """
    set_cube_pose(CUBE_HOME)


def park_cube():
    """把方块扔到机械臂够不着的地方（开局清场用）。

    为什么开局要先"扔远"再回零位：
      上一次演示要是中途失败，方块很可能被夹爪捏在手里、或者卡在指缝里 ——
      这时臂在 **Gazebo 物理里是被顶住的**。回零位的轨迹照样能规划出来，
      但实际关节根本跟不上参考，JTC 立刻报 path tolerance violation，
      MoveIt 回 CONTROL_FAILED(-4)，而报错点离真正的根因十万八千里。
      先把方块挪走，臂就自由了，演示因此能从任何烂摊子自我恢复、反复跑。
    """
    set_cube_pose(CUBE_PARK)


class PickDemo(Node):
    def __init__(self, approach_h, lift_h, use_scene, dry_run, hold):
        super().__init__('pick_demo_moveit')
        self.approach_h = approach_h
        self.lift_h = lift_h
        self.use_scene = use_scene
        self.dry_run = dry_run
        self.hold = hold

        cbg = ReentrantCallbackGroup()
        self.move_cli = ActionClient(self, MoveGroup, '/move_action', callback_group=cbg)
        self.exec_cli = ActionClient(self, ExecuteTrajectory, '/execute_trajectory',
                                     callback_group=cbg)
        self.grip_cli = ActionClient(self, GripperCommand,
                                     '/gripper_controller/gripper_cmd',
                                     callback_group=cbg)
        self.fk_cli = self.create_client(GetPositionFK, '/compute_fk', callback_group=cbg)
        self.cart_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path',
                                           callback_group=cbg)
        self.scene_cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene',
                                            callback_group=cbg)

        self.joint_pos = {}
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

    # ------------------------------------------------------------------ 基础
    def _on_js(self, msg):
        """缓存关节状态。

        注意：gz_ros2_control 把 URDF 里的 <mimic> 关节以「<名字>_mimic」发布
           （本项目实测：/joint_states 里是 left_claw_joint_mimic，
           不是 left_claw_joint）。不归一化的话，按模型里的关节名查表
           会落空，报"拿不到 /joint_states"，而话题其实好好地在发。
        """
        self.joint_pos = {}
        for name, pos in zip(msg.name, msg.position):
            key = name[:-len('_mimic')] if name.endswith('_mimic') else name
            self.joint_pos[key] = pos

    def start_executor(self):
        ex = MultiThreadedExecutor()
        ex.add_node(self)
        self._executor = ex
        threading.Thread(target=ex.spin, daemon=True).start()

    def wait(self, future, timeout, what):
        """等 future 完成，超时抛异常。用事件回调，避免嵌套 spin。"""
        ev = threading.Event()
        future.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            raise TimeoutError('%s 超时（%.0f s）' % (what, timeout))
        return future.result()

    def call(self, cli, req, timeout, what):
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('服务 %s 不可用' % what)
        return self.wait(cli.call_async(req), timeout, what)

    def send_action(self, cli, goal, timeout, what):
        if not cli.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('action %s 不可用' % what)
        handle = self.wait(cli.send_goal_async(goal), 30.0, what + '/发送')
        if handle is None or not handle.accepted:
            raise RuntimeError('%s 目标被拒绝' % what)
        return self.wait(handle.get_result_async(), timeout, what + '/执行')

    def robot_state(self):
        """用 /joint_states 拼一个 RobotState，作为规划/笛卡尔的起点。"""
        missing = [j for j in ALL_JOINTS if j not in self.joint_pos]
        if not self.joint_pos or missing:
            raise RuntimeError('拿不到 /joint_states 里的这些关节: %s'
                               % (missing or '（一条都没收到，控制器起了吗？）'))
        rs = RobotState()
        rs.joint_state.name = list(ALL_JOINTS)
        rs.joint_state.position = [float(self.joint_pos[j]) for j in ALL_JOINTS]
        rs.is_diff = False
        return rs

    def ee_pose(self):
        """当前 EE 位姿（世界系）。笛卡尔路径要拿它当起点与姿态参考。"""
        req = GetPositionFK.Request()
        req.header.frame_id = FRAME
        req.fk_link_names = [EE_LINK]
        req.robot_state = self.robot_state()
        res = self.call(self.fk_cli, req, 20.0, '/compute_fk')
        if res.error_code.val != MoveItErrorCodes.SUCCESS or not res.pose_stamped:
            raise RuntimeError('compute_fk 失败（错误码 %d）' % res.error_code.val)
        return res.pose_stamped[0].pose

    # ------------------------------------------------------------ 规划场景
    #
    # 这里要对"方块在不在 MoveIt 的世界模型里"做精细管理，原因是实测踩到的坑：
    #
    #   抓取的本质就是**手指几何压进方块**（夹持面一合上就接触/微穿透）。
    #   所以一旦夹爪闭合，只要方块还留在世界坐标系里，MoveIt 就会认为
    #   【机器人当前状态就在自碰撞】，紧接着的抬起 compute_cartesian_path
    #   直接返回 fraction = 0.000（一条路都走不了）。
    #
    #   判决实验（同一段竖直抬升）：
    #       avoid_collisions=True   → fraction = 0.000，1 个点
    #       avoid_collisions=False  → fraction = 1.000，22 个点
    #   于是确认是碰撞检测卡的，不是 IK。
    #
    # 正确解法是 MoveIt 的惯例：把抓到的物体从世界**摘下来挂到手上**
    # （AttachedCollisionObject），并声明哪些连杆允许与它接触（touch_links）。
    # 这样既不"天生碰撞"，MoveIt 也仍然知道手里握着东西。
    def _push_scene(self, world=(), attached=()):
        """下发场景增量。

        注意：ROS 2 的 moveit_msgs 里，PlanningScene 本身**没有**
           attached_collision_objects 字段 —— 它搬到了 RobotState 里
           （ROS 1 时代是在 PlanningScene 上的）。写成
           `scene.attached_collision_objects` 会 AttributeError。
        """
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = list(world)
        scene.robot_state.attached_collision_objects = list(attached)
        self.call(self.scene_cli, ApplyPlanningScene.Request(scene=scene), 20.0,
                  '/apply_planning_scene')

    def scene_add_cube(self, xyz):
        """把方块登记进世界（作为静态障碍物）。"""
        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = CUBE_ID
        co.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [CUBE_SIZE] * 3
        co.primitives = [prim]
        p = Pose()
        p.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
        p.orientation.w = 1.0
        co.primitive_poses = [p]
        self._push_scene(world=[co])

    def scene_remove_cube(self):
        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = CUBE_ID
        co.operation = CollisionObject.REMOVE
        self._push_scene(world=[co])

    def scene_attach_cube(self, xyz):
        """方块改挂到 claw_base 上（同时从世界里摘掉）。

        touch_links 里列的是"允许与方块接触"的连杆 —— 手指必须在内，
        否则挂上以后手指与方块照样算碰撞。
        """
        aco = AttachedCollisionObject()
        aco.link_name = ATTACH_LINK
        aco.touch_links = list(TOUCH_LINKS)
        aco.object.header.frame_id = FRAME
        aco.object.id = CUBE_ID
        aco.object.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [CUBE_SIZE] * 3
        aco.object.primitives = [prim]
        p = Pose()
        p.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
        p.orientation.w = 1.0
        aco.object.primitive_poses = [p]
        self._push_scene(attached=[aco])

    def scene_detach_cube(self):
        """把方块从手上摘下来（不自动放回世界，由调用方按实测位姿再登记）。"""
        aco = AttachedCollisionObject()
        aco.link_name = ATTACH_LINK
        aco.touch_links = list(TOUCH_LINKS)
        aco.object.header.frame_id = FRAME
        aco.object.id = CUBE_ID
        aco.object.operation = CollisionObject.REMOVE
        self._push_scene(attached=[aco])

    # -------------------------------------------------------------- 运动原语
    def move_joints(self, positions, label, scale=None):
        """关节空间规划 + 执行（MoveGroup action，plan_only=False）。"""
        v = VEL_SCALE if scale is None else scale
        a = ACC_SCALE if scale is None else scale
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = PLAN_ATTEMPTS
        req.allowed_planning_time = PLAN_TIME
        req.max_velocity_scaling_factor = v
        req.max_acceleration_scaling_factor = a
        cons = Constraints()
        for name, val in zip(ARM_JOINTS, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            cons.joint_constraints.append(jc)
        req.goal_constraints = [cons]
        return self._run_move(req, label)

    def move_to_position(self, x, y, z, label):
        """**位姿目标**：让 EE_LINK 的参考点落进一个半径 POS_TOL 的球里。

        这里不给姿态约束 —— 5 自由度的臂做不了完整 6D 位姿 IK
        （position_only_ik: true 就是这个意思）。
        """
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = PLAN_ATTEMPTS
        req.allowed_planning_time = PLAN_TIME
        req.max_velocity_scaling_factor = VEL_SCALE
        req.max_acceleration_scaling_factor = ACC_SCALE
        pc = PositionConstraint()
        pc.header.frame_id = FRAME
        pc.link_name = EE_LINK
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [POS_TOL]
        pc.constraint_region.primitives = [prim]
        p = Pose()
        p.position = Point(x=x, y=y, z=z)
        p.orientation.w = 1.0
        pc.constraint_region.primitive_poses = [p]
        pc.weight = 1.0
        cons = Constraints()
        cons.position_constraints = [pc]
        req.goal_constraints = [cons]
        return self._run_move(req, label)

    def _run_move(self, req, label):
        goal = MoveGroup.Goal()
        goal.request = req
        opts = PlanningOptions()
        opts.plan_only = self.dry_run
        goal.planning_options = opts
        t0 = time.time()
        res = self.send_action(self.move_cli, goal, T_EXEC_MOVE, label)
        dt = time.time() - t0
        r = res.result
        code = r.error_code.val
        ok = (code == MoveItErrorCodes.SUCCESS)
        print('      规划耗时 %.2f s   轨迹 %d 点   速度缩放 %.2f   %s总耗时 %.1f s   错误码 %d'
              % (r.planning_time,
                 len(r.planned_trajectory.joint_trajectory.points),
                 req.max_velocity_scaling_factor,
                 '（dry-run 未执行）' if self.dry_run else '', dt, code))
        return ok, code, dt

    # ------------------------------------------------------------ 笛卡尔路径
    def cartesian_to_xyz(self, x, y, z, label, exec_it=True):
        """从当前 EE 位姿出发，直线移动到目标位置 (x, y, z)。

        只改位置、**姿态原样照抄** —— 这样插值过程中姿态不变，
        不会中途拧出怪姿势导致 IK 失败。
        """
        cur = self.ee_pose()
        tgt = Pose()
        tgt.position = Point(x=float(x), y=float(y), z=float(z))
        tgt.orientation = cur.orientation

        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.start_state = self.robot_state()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK
        req.waypoints = [tgt]
        req.max_step = CART_MAX_STEP
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = CART_VEL_SCALE
        req.max_acceleration_scaling_factor = CART_ACC_SCALE

        t0 = time.time()
        res = self.call(self.cart_cli, req, 60.0, '/compute_cartesian_path')
        frac = res.fraction
        npts = len(res.solution.joint_trajectory.points)
        print('      %s   (%.4f, %.4f, %.4f) → (%.4f, %.4f, %.4f)'
              % (label, cur.position.x, cur.position.y, cur.position.z, x, y, z))
        print('      路径覆盖率 fraction = %.3f   解出 %d 点   计算 %.2f s   （速度缩放 %.2f）'
              % (frac, npts, time.time() - t0, CART_VEL_SCALE))
        if frac < 0.999:
            print('      fraction < 1.0：直线中间有段规划不出来（撞东西了？IK 失败？）')
            return False, frac

        if not exec_it or self.dry_run:
            print('      （dry-run，不执行）')
            return True, frac

        ex = ExecuteTrajectory.Goal()
        ex.trajectory = res.solution
        eres = self.send_action(self.exec_cli, ex, T_EXEC_CART, 'execute_trajectory')
        code = eres.result.error_code.val
        ok = (code == MoveItErrorCodes.SUCCESS)
        print('      执行%s   错误码 %d' % ('成功' if ok else '失败', code))
        return ok, frac

    def cartesian_to_z(self, target_z, label, exec_it=True):
        """竖直直线：x/y 保持不动，只把 z 送到 target_z（下压/抬起用）。"""
        cur = self.ee_pose()
        return self.cartesian_to_xyz(cur.position.x, cur.position.y, target_z,
                                     label, exec_it)

    def align_above_cube(self, cube_xy, tol=0.0003, max_iters=3):
        """**闭环横向对中**：把 TCP 的 x/y 修正到方块实测 x/y 上。

        为什么必须做这一步（实测数据）：
          预抓取位是「2 mm 半径的位置约束球」规划出来的，规划器一进球就停，
          实测留下约 **(+0.99, +0.47) mm** 的横向偏差。而两指与方块之间
          **每侧只有 2.14 mm 名义余量**（jaw=0 时面间距 29.28 mm，方块 25 mm）。
          于是下压时手指会擦到方块，把它**推走十几毫米**（实测 +12.08 mm），
          夹爪再闭合时两指之间已经没有东西 —— 表现为
          `指令 -0.2 → 实际 0.0000, reached_goal=False, stalled=True`。

          这个失败是**间歇的**（擦多擦少看运气），所以特别难缠。
          对中一次就好：笛卡尔移动的终点误差很小（参考停住后伺服会收敛），
          实测 1 次就够，留 3 次上限兜底。
        """
        for i in range(max_iters):
            tcp = self.ee_pose().position
            dx = cube_xy[0] - tcp.x
            dy = cube_xy[1] - tcp.y
            dist = math.hypot(dx, dy)
            print('      对中第 %d 次：TCP 偏差 dx=%+.2f dy=%+.2f mm（共 %.2f mm）'
                  % (i + 1, dx * 1000, dy * 1000, dist * 1000))
            if dist < tol:
                print('      已对准')
                return True
            ok, _ = self.cartesian_to_xyz(cube_xy[0], cube_xy[1], tcp.z,
                                          '横向对中', exec_it=True)
            if not ok:
                print('      对中移动失败')
                return False
        tcp = self.ee_pose().position
        d = math.hypot(cube_xy[0] - tcp.x, cube_xy[1] - tcp.y)
        print('      注意：对中 %d 次后仍有 %.2f mm 偏差（继续，但抓取可能不稳）'
              % (max_iters, d * 1000))
        return True

    # ----------------------------------------------------------------- 夹爪
    def gripper(self, position, label):
        if self.dry_run:
            print('      指令 %.4f rad（dry-run，不执行）' % position)
            return False, False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = GRIP_EFFORT
        res = self.send_action(self.grip_cli, goal, T_EXEC_GRIP, label)
        r = res.result
        print('      指令 %.4f rad → 实际 %.4f rad   reached_goal=%s  stalled=%s'
              % (position, r.position, r.reached_goal, r.stalled))
        if r.stalled:
            print('      （stalled = 手指压到东西停住了，抓取时正是想要的）')
        return r.reached_goal, r.stalled


def _report_time(t0):
    print('\n总耗时 %.1f s' % (time.time() - t0))
    print('  本机实测实时因子 RTF ≈ 0.39（软件渲染 + 开界面），'
          '即仿真里 1 s 的动作要花 ~2.6 s 真实时间。')
    print('  跑得更快：gz_launch 加 gui:=false 关掉界面，或 camera:=false 关掉相机。')
    print('=' * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只规划不执行')
    ap.add_argument('--no-scene', action='store_true', help='不把方块登记进规划场景')
    ap.add_argument('--hold', action='store_true', help='抬起后停住，不放回（截图用）')
    ap.add_argument('--approach', type=float, default=0.060, help='预抓取高度 (m)')
    ap.add_argument('--lift', type=float, default=0.130, help='抬起高度 (m)')
    ap.add_argument('--no-reset', dest='reset_cube', action='store_false',
                    help='不复位方块（默认会先复位到初始位姿）')
    args = ap.parse_args()

    rclpy.init()
    node = PickDemo(args.approach, args.lift, not args.no_scene,
                    args.dry_run, args.hold)
    node.start_executor()
    t_start = time.time()

    print('=' * 72)
    print('MoveIt 2 抓取闭环演示' + ('    [DRY-RUN：只规划不执行]' if args.dry_run else ''))
    print('=' * 72)

    try:
        # ---------- 0. 清掉上次残留的场景 + 把臂挪到零位 ----------
        # 注意：这两步不能省，也不能和下一步换顺序。
        #    ① 先摘掉场景里可能残留的方块：如果臂此刻正夹着它，MoveIt 会认为
        #       起始状态就在碰撞中，连规划都起不来。
        #    ② 复位方块用的是 Gazebo 的 set_pose（瞬移）。如果臂此刻正夹着
        #       方块，瞬移会把它直接塞进手指的碰撞体里，接触解算爆出巨大冲击力
        #       把整条臂弹开 —— 表现为 JTC 报 "path tolerance violation"、
        #       MoveIt 回 CONTROL_FAILED(-4)，而报错点离真正的根因很远。
        #    先清场景、再把臂挪开，这个演示就能反复跑。
        print('[0] 清残留场景 + 方块挪走 + MoveIt 规划回零位')
        if node.use_scene:
            node.scene_remove_cube()
        park_cube()
        time.sleep(1.5)
        # 恢复步用**低速**（HOME_VEL_SCALE）：上次要是半路失败，臂可能正顶着
        # 地面/方块，低速时伺服滞后小、容差余量大，能慢慢"磨"出来；速度高了会
        # 直接报 CONTROL_FAILED。正常起步（臂本来就在零位附近）这一步几乎瞬间完成。
        ok, code, _ = node.move_joints([0.0] * 5, 'move_home', scale=HOME_VEL_SCALE)
        if not ok:
            print('    回零位失败（错误码 %d）' % code)
            print('       臂可能卡住了（顶着地面或工件）。最省事的办法是重启仿真：')
            print('         ros2 launch arm_gazebo gz_launch.py')
            return 1
        print()

        # ---------- 1. 复位方块 + 读位姿，推出抓取目标 ----------
        if args.reset_cube:
            reset_cube()
            time.sleep(2.5)
            print('[1] 方块已复位到世界文件里的初始位姿')
        cx, cy, cz = read_cube_pose()
        print('[1] 方块实测位姿        x=%+.4f  y=%+.4f  z=%.4f' % (cx, cy, cz))
        grasp_z = GRASP_TCP_Z
        pre_z = grasp_z + args.approach
        lift_z = grasp_z + args.lift
        print('    抓取目标 TCP        (%+.4f, %+.4f, %.4f)   ← 方块 XY + 指尖间隙推的高度'
              % (cx, cy, grasp_z))
        print('    预抓取 TCP z        %.4f   (抓取位上方 %.0f mm)'
              % (pre_z, args.approach * 1000))
        print('    抬起目标 TCP z      %.4f   (抓取位上方 %.0f mm)'
              % (lift_z, args.lift * 1000))
        print('    注意：位姿目标只约束位置：5 自由度的姿态自由度由 KDL 自行分配')

        if node.use_scene:
            node.scene_add_cube((cx, cy, cz))
            print('    规划场景            方块 25 mm 立方体已登记为障碍物')
        else:
            print('    规划场景            跳过（--no-scene）')
        print()

        # ---------- 2. 张开夹爪 ----------
        print('[2] 张开夹爪')
        node.gripper(GRIP_OPEN, 'gripper/open')
        print()

        # ---------- 3. MoveIt 位姿目标 → 预抓取位 ----------
        print('[3] MoveIt 规划到预抓取位（方块正上方 %.0f mm，位姿目标）'
              % (args.approach * 1000))
        ok, code, _ = node.move_to_position(cx, cy, pre_z, 'move_to_pregrasp')
        if not ok:
            print('    规划到预抓取位失败（错误码 %d，见 MoveItErrorCodes）' % code)
            return 1
        print('    到位')

        # ---------- 3.5 闭环横向对中 ----------
        # 预抓取位是用「2 mm 半径的球」当目标的，规划器一进球就停，
        # 实测会留下约 1 mm 横向偏差；而手指与方块每侧只有 2.14 mm 余量。
        # 不校正的话下压会把方块推走（实测 +12 mm），夹爪就合不上了。
        print('[3.5] 闭环横向对中（把 TCP 的 x/y 修正到方块中心）')
        if not node.align_above_cube((cx, cy)):
            print('    对中失败')
            return 1
        print()

        # ---------- 4. 笛卡尔竖直下压 ----------
        print('[4] 笛卡尔直线下压（竖直进刀，不是关节空间圆弧）')
        ok, _ = node.cartesian_to_z(grasp_z, '下压')
        if not ok:
            print('    下压失败')
            return 1
        print('    到位，指尖应悬在方块两侧')
        print()

        # ---------- 5. 闭合 ----------
        print('[5] 闭合夹爪')
        node.gripper(GRIP_CLOSE, 'gripper/close')
        p_close = read_cube_pose()
        print('    闭合后方块 z = %.4f（应仍在 %.4f 附近）' % (p_close[2], cz))

        # 方块从"世界里的障碍物"变成"手里的东西"。
        # 不做这步的话，接下来的抬起会 fraction=0 —— 手指与方块几何重叠，
        # MoveIt 认为当前状态就是碰撞。详见 scene_attach_cube 的注释。
        if node.use_scene and not args.dry_run:
            node.scene_attach_cube(p_close)
            print('    规划场景            方块已挂到 %s 上（touch_links: %s）'
                  % (ATTACH_LINK, ', '.join(TOUCH_LINKS)))
        print()

        # ---------- 6. 笛卡尔竖直抬起 ----------
        print('[6] 笛卡尔直线抬起（夹爪全程朝下）')
        ok, _ = node.cartesian_to_z(lift_z, '抬起')
        if not ok:
            print('    抬起失败')
            return 1
        print('    到位')
        time.sleep(3)

        # ---------- 7. 判定 ----------
        p1 = read_cube_pose()
        dz = p1[2] - cz
        print()
        print('-' * 72)
        print('抬起后方块位姿      x=%+.4f  y=%+.4f  z=%.4f' % (p1[0], p1[1], p1[2]))
        print('相对初始抬升        Δz = %+.1f mm' % (dz * 1000))
        grasped = dz > CUBE_LIFT_THRESHOLD
        if grasped:
            print('抓取成功：方块被提离地面 %.1f mm' % (dz * 1000))
        else:
            print('抓取失败：方块基本没动（Δz = %.1f mm）' % (dz * 1000))
            print('   可能原因：夹持力不足（滑脱）／位形没对准／下压撞到方块')
        print('-' * 72)

        if args.hold:
            print('\n--hold：停在此姿态不退场，方便截图。')
            _report_time(t_start)
            return 0

        if not grasped or args.dry_run:
            print('\n未成功抓取或处于 dry-run，跳过放回流程。')
            _report_time(t_start)
            return 0

        # ---------- 8. 放回 ----------
        print()
        print('[8] 放回：下压到「方块最低点离地 %.0f mm」→ 张开 → 撤出'
              % (PLACE_CLEARANCE * 1000))
        # 下压终点**按方块实测姿态算**，而不是让 TCP 回到抓起时的高度。
        # 推导：方块此刻中心在 cube_z，姿态决定它沿世界 Z 的支撑半长 support，
        #       于是最低点 = cube_z - support；要把它降到离地 PLACE_CLEARANCE，
        #       还要往下 drop = (cube_z - support) - PLACE_CLEARANCE，
        #       对应 TCP 终点 = 当前 TCP z - drop。
        cube_now, cube_rpy = read_cube_pose_full()
        support = cube_support_half_z(cube_rpy)
        tcp_now = node.ee_pose().position.z
        drop = (cube_now[2] - support) - PLACE_CLEARANCE
        place_tcp_z = tcp_now - drop
        print('    方块中心 z=%.4f   姿态 rpy=(%+.3f, %+.3f, %+.3f)'
              % (cube_now[2], cube_rpy[0], cube_rpy[1], cube_rpy[2]))
        print('    沿 Z 支撑半长 %.2f mm  →  最低点 z=%+.4f'
              % (support * 1000, cube_now[2] - support))
        print('    TCP 现在 %.4f，再降 %.1f mm 到 %.4f'
              % (tcp_now, drop * 1000, place_tcp_z))
        ok, _ = node.cartesian_to_z(place_tcp_z, '下压放回')
        if not ok:
            print('    放回下压失败')
            return 1
        node.gripper(GRIP_OPEN, 'gripper/open')
        time.sleep(2)
        p_release = read_cube_pose()
        # 松手了：方块重新变成"世界里的障碍物"，按**实测**位姿登记回去
        if node.use_scene:
            node.scene_detach_cube()
            node.scene_add_cube(p_release)
            print('    规划场景            方块已摘下手、按实测位姿重新登记为地面障碍物')
        node.cartesian_to_z(lift_z, '抬起撤离')

        p2 = read_cube_pose()
        print('    放回后方块位姿  x=%+.4f  y=%+.4f  z=%.4f' % (p2[0], p2[1], p2[2]))
        print('    与原位误差      Δx=%+.2f mm  Δy=%+.2f mm  Δz=%+.2f mm'
              % ((p2[0] - cx) * 1000, (p2[1] - cy) * 1000, (p2[2] - cz) * 1000))

        # ---------- 9. 回零位 ----------
        # 注意：必须用 HOME_VEL_SCALE，不能用默认的 VEL_SCALE ——
        #    这是全程最大的一次关节空间移动，0.40 会顶穿跟踪容差被杀。
        #    详见文件头常量处的实测记录。
        print()
        print('[9] MoveIt 规划回零位（速度缩放 %.2f）' % HOME_VEL_SCALE)
        ok, code, _ = node.move_joints([0.0] * 5, 'move_home', scale=HOME_VEL_SCALE)
        print('    %s' % ('到位' if ok else '失败（错误码 %d）' % code))

        # ---------- 10. 收尾：撤掉规划场景里的方块 ----------
        if node.use_scene:
            node.scene_remove_cube()
            print('[10] 规划场景           方块已撤出')

        _report_time(t_start)
        return 0

    except (TimeoutError, RuntimeError) as e:
        print('\n错误：%s' % e)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
