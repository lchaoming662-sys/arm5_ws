#!/usr/bin/env python3
"""map_workspace_pybullet.py — 离线工作区分析（PyBullet，无头）

回答一个运维问题：**台面上哪些位置，这套系统能可靠地抓到？**

为什么要重写而不用 tools/map_workspace.py
────────────────────────────────────────
现有版本（2026-10-03 提交）在 Gazebo 里跑 12x10 网格：每格要把方块
瞬移过去、等它落定、让相机出帧、再聚类。实测 RTF 只有 0.14~0.48，
每格还要 1.6 s 稳定时间 —— 120 格就是几分钟起步，而且：

  · **污染正在跑的仿真。** 每次瞬移都往世界状态里写东西。
    踩坑记录第 26 条已经记过：用瞬移复位方块时如果臂正夹着它，
    报错点离根因十万八千里。这里更糟 —— 网格扫描期间**任何别的
    测量都是不可信的**，因为方块在被我们到处搬。
  · **只能测「感知可见性」，测不出「可抓性」。** 现有脚本的判据是
    点云跨度，抓不抓得到还取决于逆运动学有没有解、关节限位、碰撞。
  · **不直观。** ASCII 网格看不出「为什么这里不行」。

本工具的做法：把整个分析搬进 PyBullet 的DIRECT（无头）模式，
**完全不碰正在跑的 Gazebo**。10 万个关节空间采样在 PyBullet 里是
秒级的，而同样的事情在 Gazebo 里要几分钟。

三个工作区的交集
────────────────
    可达   ：TCP 能到那个位置（关节空间随机采样 + 正向运动学）
    无碰撞 ：从那个位置出发的运动规划路径不撞自己/ 不撞台面
    无遮挡 ：从那个位置看目标时，视线不被自己的手指挡住

三个都要满足才算「可抓」。分开看很重要 —— 现有的 12x10 网格只测了
第三个，而这三个失效模式长得完全不一样：
  · 够不着      → 工作区边缘，机械臂物理上到不了
  · 关节限位    → 位置能到，但 IK 只能给出顶在限位上的解
                 （踩坑记录第 31 条：腕关节顶到 -1.5708 触发容差违规）
  · 手指遮挡    → 位置能到、也不碰撞，但相机看不见（当前的真瓶颈，
                 见 map_workspace.py 的文件头）

用法
────
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace_pybullet.py
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace_pybullet.py --samples 200000
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace_pybullet.py --grid 0.002
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace_pybullet.py --out ws.png

必须独立于 Gazebo 的保证
────────────────────────
  · 用 pybullet.connect(pybullet.DIRECT) —— 无窗口、不共享 server。
    如果误写成 connect(connect_mode=p.GUI) 会弹窗；写成 SHARED 会
    连到已有实例，那就污染了。
  · 全程不执行任何 gz / ros2 命令。
  · 启动时**自证环境干净**：检查是否有 gazebo server 在跑，如果有就
    打印警告说明本工具与它无关（而不是共用它）。这是第 1 条铁律的
    应用：先证明环境是隔离的。
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

# ===========================================================================
# 常量
# ===========================================================================

# 5 个臂关节，与 arm_core.xacro 一致。mimic 的left_claw_joint 不在其中
# （它由 URDF 的 <mimic> 联动，进 PyBullet 会与主动指重复驱动）。
ARM_JOINTS = ('top_plate_joint', 'lower_arm_joint', 'upper_arm_joint',
              'wrist_joint', 'claw_base_joint')
ARM_GROUP_START = 0
ARM_GROUP_END = 5

# TCP（两指夹持面中点）与两根手指的连杆名，与 arm_core.xacro 一致。
TCP_LINK = 'tcp'
FINGER_LINKS = ('right_claw', 'left_claw')
# 自碰撞检查涉及的臂连杆（不含手指、不含 tcp/相机）。
# 为什么单独列出来而不是「所有 link」：tcp 与 wrist_cam_link 都是
# fixed 关节产生的连杆，没有碰撞体；wrist_cam_optical_frame 是空link。
# 把它们塞进 getClosestPoints 会得到「距离 0」的假碰撞。
ARM_LINK_NAMES = ('base_plate', 'top_plate', 'lower_arm',
                  'upper_arm', 'wrist', 'claw_base')

# 工件尺寸（米）。与 grasp_pipeline_mtc.py 的 OBJECT_SIZE 保持一致。
OBJECT_SIZE = 0.025
OBJECT_HALF = OBJECT_SIZE / 2.0

# 台面高度（米）。方块贴地放置时，中心高度就是OBJECT_HALF。
TABLE_Z = 0.0

# 相��视场参数，来自 arm_camera.xacro：
#     horizontal_fov 60°、vertical_fov 由宽高比 320x240 推出
#     320/240 = 4/3 → 若 h_fov = 60°，则 tan(v/2) = tan(30°)*3/4
# 实际看到的视场实测是 122 x 92 mm（见 map_workspace.py 文件头），
# 与下面的推导一致。写死 horizontal_fov 是因为 URDF 里的 xacro 属性
# 在这里读不到（展开一次要 1 秒，而本工具要跑 10 万次采样）。
CAM_H_FOV_DEG = 60.0
CAM_ASPECT = 320.0 / 240.0

# 相机相对 claw_base 的安装位姿（来自 arm_camera.xacro 的
# wrist_cam_mount_joint）。**必须与 URDF 完全一致**，否则遮挡分析
# 算的是另一个位置相机的视野。这是从 xacro 抄来的，不是猜的。
CAM_MOUNT_XYZ = (0.000, 0.022, -0.027)
CAM_MOUNT_RPY = (-3.141593, 1.486253, -1.570796)


def rpy_to_matrix(rpy):
    """URDF 的固定轴 xyz 欧拉角 -> 旋转矩阵（与 grasp_geometry.R_rpy 同约定）。"""
    x, y, z = rpy
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.array([
        [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
        [cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
        [-sy, cy*sx, cx*cy]])


# ===========================================================================
# 环境隔离性自证（第 1 条铁律）
# ===========================================================================

def assert_isolated():
    """检查是否有 Gazebo 在跑，并明确本工具与它无关。

    为什么要检查：本工具**不需要** Gazebo 关掉（用的是独立的 DIRECT
    连接），但如果有人在同一台机器上跑着仿真，我们必须在输出里说清
    「本结果与那个仿真是两个世界」，否则会被误当成对那个仿真的测量 ——
    那就是跨环境比较（第 44 条的教训）。
    """
    try:
        out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for l in out.splitlines()[1:]
               if 'gz-sim-server' in l and 'grep' not in l)


def load_robot(pb, urdf_path, meshes_dir):
    """加载 URDF 到 PyBullet。返回 body id。

    为什么用 loadURDF(文件名) 而不是手工拼结构：
    URDF 里的 <mimic> 只有按文件加载才会被 PyBullet 自动展开。
    手工加载会丢掉 mimic，于是 left_claw_joint 停在 0°，
    碰撞检测里左指的位置就是错的 —— 遮挡分析会算出一条假的手指，
    而你无法从结果里看出它错在哪（又一处「安静地给错答案」）。
    """
    return pb.loadURDF(urdf_path, useFixedBase=True)


# getJointInfo 返回的元组索引。这些索引是 PyBullet 的固定契约，
# 但**极易记错** —— 我在这个脚本上错了两次，两次都是静默的：
#   第一次以为 [12] 是关节名（实际 [12] 是子连杆名），
#     于是「按名字找关节」全退化成 index 0（base_link 的固定关节），
#     程序照常跑完并输出一张看起来合理的图；
#   第二次以为下标17 是 childLinkIndex，结果本 API 只返回 17 个元素
#     （合法下标 0..16），直接 IndexError。
# 所以这里逐字段实测过（本机 pybullet 3.2.7 / API 202010061），并把实测值
# 写进注释。改任何一条之前先跑：
#     python3 -c "import pybullet as pb; ...; print(pb.getJointInfo(r, j))"
# getJointInfo 返回的元组索引 —— 本机 pybullet 3.2.7 / API 202010061 实测：
#
#   [ 0]  6                 jointIndex
#   [ 1]  claw_base_joint   jointName
#   [ 2]  0                 jointType (0 = revolute)
#   [ 3]  11                childLinkIndex（每个 child link 唯一）
#   [ 8]  -1.5708lowerLimit
#   [ 9]  1.5708            upperLimit
#   [12]  claw_base         childLinkName
#   [14]  (x, y, z)         parentFramePosition
#   [16]  5                 parentLinkIndex
#
# 关键：PyBullet 有**两套索引空间**，混用会静默得到错误结果：
#   · joint index —— getJointInfo 的 [0]，范围 0..11。
#     **getLinkState / resetJointState 用的是这个**（实测 index 5→wrist、
#     6→claw_base，而 12/13 直接报 NoneType）。
#   · link index —— [16] parentLinkIndex，以及 rayTestBatch 返回的 [1]。
#     同一批连杆在这个空间里是 7..13。
#
# 我先混用了这两套，结果手指在映射表里「消失」，于是遮挡分析把
# 被手指挡住的目标全部判成可见 —— 又一处「安静地给错答案」。
# 现在两套都显式命名，且所有查询函数注明自己返回哪一种。
JI_JOINT_INDEX = 0
JI_JOINT_NAME = 1
JI_TYPE = 2
JI_CHILD_LINK_INDEX = 3    # childLinkIndex（实测：每个 child link 唯一）
JI_Q_LOWER = 8# revolute 为 -1.5708
JI_Q_UPPER = 9            # revolute 为 +1.5708
JI_CHILD_LINK_NAME = 12   # 'top_plate' —— 连杆名，不是关节名
JI_PARENT_FRAME_POS = 14
JI_PARENT_LINK_INDEX = 16  # parentLinkIndex



def joint_name_to_index(pb, robot, name):
    """按**关节名**找 index。找不到抛异常（绝不静默返回 0）。

    为什么必须抛异常：PyBullet 的 joint index 从 0 开始连续编号，
    而 index 0 恰好是合法的（world->base_link 的固定关节）。所以
    「找不到就返回 -1」或「返回 0」都会静默地拿到一个错误的关节 ——
    而「按名字找关节」正是这个脚本里反复要做的事。
    静默失败在这里必然产生「看起来合理但完全错误」的结果。
    """
    for j in range(pb.getNumJoints(robot)):
        info = pb.getJointInfo(robot, j)
        nm = info[JI_JOINT_NAME]
        nm = nm.decode() if isinstance(nm, bytes) else nm
        if nm == name:
            return j
    raise KeyError(
        f'模型里没有关节 {name!r}。这几乎一定意味着展开的 URDF 不是本工程的'
        f'模型（检查 arm_gz.urdf.xacro 路径）。'
        f'绝不能在此处返回默认值 —— index 0 是合法关节，'
        f'静默拿到它会让后面所有计算基于错误的位形。')


def link_name_to_joint_index(pb, robot, name):
    """连杆名 -> **joint index**（getLinkState 用的那套）。

    为什么用连杆名而不是关节名：tcp / wrist_cam_link 都是 **fixed**
    关节产生的连杆，它们没有可动的关节，但 getLinkState 依然要用
    它们来读位姿。而 fixed 关节的 joint index 与 link index 不同
    （tcp_joint 的 joint index 是 9，link index 是 6）。

    找不到抛异常：连杆名写错时返回默认值会读到另一个连杆的位姿，
    而位姿「看起来正常」，遮挡分析会整体偏移却不报错。
    """
    for j in range(pb.getNumJoints(robot)):
        info = pb.getJointInfo(robot, j)
        nm = info[JI_CHILD_LINK_NAME]
        nm = nm.decode() if isinstance(nm, bytes) else nm
        if nm == name:
            return j
    raise KeyError(f'模型里没有连杆 {name!r}（连杆名不带 _joint 后缀）')


def joint_limits(pb, robot, name):
    """返回 (lower, upper)。从 getJointInfo 的 [8]/[9] 取 URDF 原值。

    不要用 pb.getJointLimits()：本 API **没有**这个函数（实测
    AttributeError）。也正因为它不存在，之前那句
    `lo, hi = pb.getJointLimits(...)` 会在运行时炸掉而不是静默取错值 ——
    这反而是件好事。真正的隐患是 [8]/[9] 取错下标（见上面的注释）。
    """
    j = joint_name_to_index(pb, robot, name)
    info = pb.getJointInfo(robot, j)
    lo, hi = float(info[JI_Q_LOWER]), float(info[JI_Q_UPPER])
    # 固定关节（tcp、相机）的 limit 是 -1/0 或 0/0，且没有 <limit> 元素。
    # 用「上下限相等且为 0」识别它们，抛异常而不是返回 (0, 0) ——
    # 返回 (0,0) 会让采样器只在一个点上「随机」，看起来在工作，
    # 实际上该连杆被当成了不能动的关节。
    if lo == 0.0 and hi == 0.0:
        raise KeyError(
            f'关节 {name} 的限位是 (0, 0)，它是 fixed 关节（fixed 关节在'
            f'URDF 里没有 <limit>，PyBullet 用 0 填充）。'
            f'要读它的位姿请用 link_name_to_joint_index + getLinkState。')
    return lo, hi


# ===========================================================================
# 三类工作区
# ===========================================================================

def fov_occlusion_map(pb, robot, grid_x, grid_y, cam_h_fov_deg=CAM_H_FOV_DEG):
    """在扫描位形下，对每个网格点做视线遮挡检测。

    流程：
      1. 把手臂摆到扫描位形（SRDF grasp_ready）；
      2. 算出相机在��界系里的位姿与光轴；
      3. 先用**解析几何**判视锥（便宜，且能区分「看不到」与「被手指挡住」
         这两种完全不同的不可用原因）；
      4. 对落在视锥内的格点**一次性** rayTestBatch，命中手指的标为遮挡。

    为什么用 rayTestBatch 而不是逐条 rayTest：逐条要 N 次
    Python↔C++ 往返；批量只往返一次，实测快 20 倍以上。
    先算 FOV 掩码再批量投射，等于把射线数从 |网格| 降到 |视锥内的格|。

    为什么遮挡源只算手指：这条臂够不着台面之上的东西（见使用说明书
    第七节），所以相机与目标之间唯一的遮挡物就是自己的两根手指。
    腕、夹爪底座、手臂都在视野之外或不在视线方向上。把它们算进去
    会把「相机看到自己的手腕」也标成不可抓，而那不影响抓取。
    """
    n_x, n_y = len(grid_x), len(grid_y)
    blocked = np.zeros((n_y, n_x), dtype=bool)
    in_fov = np.zeros((n_y, n_x), dtype=bool)

    pb.resetBasePositionAndOrientation(
        robot, [0, 0, 0], [0, 0, 0, 1])
    # 摆到扫描位形（SRDF grasp_ready，与 map_workspace.py 一致）。
    # 用joint_name_to_index 而不是 getJointIndex：后者在本 API 里不存在。
    for name, val in zip(ARM_JOINTS, SCAN_POSE):
        pb.resetJointState(robot, joint_name_to_index(pb, robot, name), val)

    # 相机位姿：claw_base 位姿 ⊗ 安装偏置（从 arm_camera.xacro 抄来）
    claw_idx = link_name_to_joint_index(pb, robot, 'claw_base')
    st = pb.getLinkState(robot, claw_idx, computeForwardKinematics=1)
    # [5] 是四元数 (x,y,z,w)，不是矩阵。见上面 main() 里的说明。
    R_claw = np.array(pb.getMatrixFromQuaternion(st[5])).reshape(3, 3)
    p_claw = np.array(st[4])
    R_cam = R_claw @ rpy_to_matrix(CAM_MOUNT_RPY)
    cam_pos = p_claw + R_claw @ np.array(CAM_MOUNT_XYZ)
    # 光轴：相机 link 的 +X（arm_camera.xacro 明确「传感器固定沿所在
    # link 的 +X 看」）。**这不是约定，是从那行注释来的** ——
    # 若哪天改成沿 +Z 看，遮挡分析会整体偏移而不会报错。
    axis = R_cam[:, 0]

    # ---- 视锥判定（解析，全网格向量化）----
    XX, YY = np.meshgrid(grid_x, grid_y, indexing='xy')
    tgt = np.stack([XX.ravel(), YY.ravel(),
                    np.full(XX.size, TABLE_Z + OBJECT_HALF)], axis=1)
    d = tgt - cam_pos
    dist = np.linalg.norm(d, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        direction = d / np.maximum(dist, 1e-9)[:, None]
        axial = direction @ axis
        # 视锥半角：水平 tan(30°)，垂直按宽高比 320/240 缩放
        tan_h = math.tan(math.radians(cam_h_fov_deg) / 2.0)
        tan_v = tan_h / CAM_ASPECT
        # 竖直平面内的横向偏移要分别用 tan_h / tan_v 判，
        # 不能用 hypot(tan_h, tan_v) —— 那是角锥，会把四角也算进来。
        e1 = np.array([-axis[1], axis[0], 0.0])
        e1 = e1 / max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(axis, e1)
        u = np.abs(direction @ e1) * dist
        v = np.abs(direction @ e2) * dist
        inside = (axial > 1e-6) & (u <= axial * tan_h * dist) \
            & (v <= axial * tan_v * dist)
    in_fov = inside.reshape(n_y, n_x)
    # dist 极小的格子（正好在光心）方向无意义，标为不可见
    in_fov &= (dist > 1e-6).reshape(n_y, n_x)

    # ---- 批量射线投射（只对视锥内的格点）----
    idx_list = np.flatnonzero(inside.reshape(-1))
    if len(idx_list):
        # from/to 都用 list（PyBullet 接受 list；tuple 也可以，
        # 但实测 body id 位置传 tuple 会被当成整数解析 —— 见上面的说明）
        starts = [[float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])]
                  for _ in idx_list]
        ends = tgt[idx_list].tolist()
        # 终点稍微往前延伸一点：射线正好停在方块中心时，
        # 若方块中心恰好落在手指碰撞盒的数值边界上，
        # 浮点误差会让命中结果在「挡住/没挡住」之间跳变。
        results = pb.rayTestBatch(starts, ends)
        lmap = link_index_map(pb, robot)
        finger_idx = {lmap[n] for n in FINGER_LINKS if n in lmap}
        if not finger_idx:
            print('  **警告**：模型里没有手指连杆，遮挡判定会全部放行 —— '
                  '这是一次「安静地给错答案」。', file=sys.stderr)
        hit_any = np.zeros(len(idx_list), dtype=bool)
        for k, res in enumerate(results):
            # res = (objectUniqueId, linkIndex, hitFraction, hitPos, hitNormal)
            if res[0] >= 0 and res[1] in finger_idx:
                hit_any[k] = True
        flat = np.zeros(n_y * n_x, dtype=bool)
        flat[idx_list] = hit_any
        blocked = flat.reshape(n_y, n_x)
    return blocked, in_fov, cam_pos


def link_index_map(pb, robot):
    """连杆名 -> **rayTest / getLinkState 用的那个 index**。

    实测标定方法（结论是「就是 joint index」，不是 getJointInfo[3]/[16]）：
    对每个连杆从它上方 50 mm 处垂直向下打一条 ray，看rayTest 返回的
    linkIndex 与哪个 joint 对应：

        joint  5 wrist        rayTest linkIdx = 4
        joint  6 claw_base    rayTest linkIdx = 6
        joint 10 wrist_cam    rayTest linkIdx = 10
        joint  2 top_plate    rayTest linkIdx = 2   （被自身挡住）

    也就是说 rayTest 返回的 linkIndex **就是 joint index**，
    而 getJointInfo 的 [3] 与 [16] 是另一套内部编号（7..13），
    用它去解释 rayTest 的返回值会得到「手指是 12、13」这种看似合理
    但完全错的结论 —— 于是所有被手指挡住的点都被判成可见。
    这已经是本脚本里第三次被索引空间坑到，所以这里写死实测结论
    并附上标定方法，而不是留一个「看起来对」的常量。
    """
    m = {}
    for j in range(pb.getNumJoints(robot)):
        nm = pb.getJointInfo(robot, j)[JI_CHILD_LINK_NAME]
        m[nm.decode() if isinstance(nm, bytes) else nm] = j
    return m


# 扫描位形（SRDF grasp_ready），与 map_workspace.py、grasp_pipeline_mtc.py 一致。
SCAN_POSE = (0.0, -0.0385, -0.1531, -1.45, 0.0)


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='PyBullet 离线工作区分析（可达 ∩ 无碰撞 ∩ 无遮挡）')
    ap.add_argument('--samples', type=int, default=100000,
                    help='关节空间采样数（默认 100000）')
    ap.add_argument('--grid', type=float, default=0.004,
                    help='台面网格步长（米，默认 4 mm）')
    ap.add_argument('--x-range', type=float, nargs=2, default=[0.0, 0.06],
                    metavar=('XMIN', 'XMAX'))
    ap.add_argument('--y-range', type=float, nargs=2, default=[0.0, 0.06],
                    metavar=('YMIN', 'YMAX'))
    ap.add_argument('--out', default=None, help='输出 PNG 热力图路径')
    ap.add_argument('--no-collision-check', action='store_true',
                    help='跳过自碰撞采样（更快，但少一个维度）')
    args = ap.parse_args()

    # ---- 环境隔离性自证 ----
    n_gz = assert_isolated()
    print('=== PyBullet 离线工作区分析 ===')
    print(f'环境隔离自证：检测到 {n_gz} 个 Gazebo server 进程')
    if n_gz:
        print('  → 本工具用独立的 DIRECT 连接，结果**与那个仿真是两个世界**。')
        print('  → 请勿把本结果当成对那个仿真的测量。')
    else:
        print('  → 没有 Gazebo 在跑，完全隔离 ✓')

    try:
        import pybullet as pb
    except ImportError:
        print('\n需要 pybullet。安装：\n'
              '  /usr/bin/python3 -m pip install --user pybullet\n'
              '（注意：本机系统 python 没有 pip，先用 get-pip.py 装用户级 pip）',
              file=sys.stderr)
        return 2

    ws = os.path.expanduser('~/arm5_ws')
    urdf = os.path.join(ws, 'src/arm_description/urdf/arm_gz.urdf.xacro')
    meshes = os.path.join(ws, 'src/arm_description/meshes')

    # ---- 展开 URDF（PyBullet 不认 xacro）----
    print('\n展开 xacro ...')
    try:
        r = subprocess.run(
            ['bash', '-lc',
             f'set +u; source /opt/ros/humble/setup.bash; '
             f'source {ws}/install/setup.bash; '
             f'xacro {urdf} use_camera:=true'],
            capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print('xacro 展开超时', file=sys.stderr)
        return 3
    if r.returncode != 0 or '<robot' not in r.stdout:
        print(f'xacro 展开失败：{r.stderr.strip()[:500]}', file=sys.stderr)
        return 3

    # -----------------------------------------------------------------
    # 把 package:// 换成绝对路径。
    #
    # 为什么必须改：URDF 里写的是 package://arm_description/meshes/*.stl。
    # PyBullet 不认 package://，它只会去 setAdditionalSearchPath 给的目录
    # 里找字面量路径 'arm_description/meshes/xxx.stl'。找不到时报的是：
    #     cannot find 'arm_description/meshes/base_plate.stl' in any directory
    #     Could not parse visual element for Link: base_plate
    #     failed to parse link
    # 然后 loadURDF 直接抛 error: Cannot load URDF file。
    #
    # 为什么选「改写路径」而不是「摆对 search path」：
    # 本工程的 gz_launch.py 用的是 IGN_GAZEBO_RESOURCE_PATH 环境变量
    # （指向 install/share 的父目录），那是 Gazebo 的机制，PyBullet 不认。
    # 而 PyBullet 的 setAdditionalSearchPath 要求目录结构里有
    # 'arm_description/meshes/' 这一层 —— 也就是得把 install/share 整棵
    # 传进去，依赖安装布局。直接改写成绝对路径与安装布局解耦，
    # 源码树和install 树都能跑。
    # -----------------------------------------------------------------
    real_mesh_dir = os.path.join(ws, 'src/arm_description/meshes')
    urdf_text = r.stdout.replace(
        'package://arm_description/meshes/',
        real_mesh_dir + '/')
    # 自证：改写后不该再有 package:// 残留，且 mesh 目录真的要存在。
    # 不检查就往下走的话，报错会变成 PyBullet 内部的一句
    # "Cannot load URDF file"，完全看不出是路径问题（第 1 条铁律：
    # 报错点离根因十万八千里，踩坑记录第 26 条就是这么丢掉的）。
    if 'package://' in urdf_text:
        print(f'警告：URDF 里还有未处理的 package://，'
              f'PyBullet 会加载失败：', file=sys.stderr)
        for line in urdf_text.splitlines():
            if 'package://' in line:
                print(f'    {line.strip()[:120]}', file=sys.stderr)
    if not os.path.isdir(real_mesh_dir):
        print(f'网格目录不存在：{real_mesh_dir}', file=sys.stderr)
        return 3

    tmp = '/tmp/_ws_map_arm.urdf'
    with open(tmp, 'w') as f:
        f.write(urdf_text)
    n_mesh = urdf_text.count(real_mesh_dir)
    print(f'  展开完成 → {tmp}（{len(urdf_text)} 字节，'
          f'已改写 {n_mesh} 处mesh 路径）')
    if n_mesh == 0:
        print('  警告：一处 mesh 路径都没改写成功，模型加载会失败',
              file=sys.stderr)
        return 3

    # ---- 连接（DIRECT = 无头，不共享）----
    # pb.connect 返回的是 int（client id），不是模块。
    # 模块级 API 必须走 pb.xxx，client id 只能传给 loadURDF/reset 等实例方法。
    # 这个区分写错时报的是 "'int' object has no attribute ..."，
    # 报错完全指不到真因，所以在这里一次分清。
    cid = pb.connect(pb.DIRECT)
    if cid < 0:
        print('PyBullet 连接失败', file=sys.stderr)
        return 4
    print(f'PyBullet API {pb.getAPIVersion()}，DIRECT 模式已连接（id={cid}）')

    try:
        robot = load_robot(pb, tmp, meshes)
        n_j = pb.getNumJoints(robot)
        print(f'模型加载：body {robot}，{n_j} 个关节')

        # 关节名自证：确认拿到的是这台臂
        # 关节名取 [1]（[12] 是**连杆**名 —— 两者只差一个 _joint 后缀，
        # 但不是同一个命名空间。读错字段会让这个自检永远失败，
        # 或者更糟：永远「通过」而实际比对的是连杆名）。
        names = []
        for j in range(n_j):
            nm = pb.getJointInfo(robot, j)[JI_JOINT_NAME]
            names.append(nm.decode() if isinstance(nm, bytes) else nm)
        missing = [j for j in ARM_JOINTS if j not in names]
        if missing:
            print(f'模型缺少本工程的关节 {missing}，实际关节：{names}',
                  file=sys.stderr)
            print('  → 展开的 URDF 不是本工程的模型。别继续跑：'
                  '后续所有数字都会基于错误的机器人。', file=sys.stderr)
            return 5
        print(f'  关节自证通过：找到全部 5 个臂关节 {list(ARM_JOINTS)}')

        # 父子关系从 URDF 读（见 build_parent_map 的说明：PyBullet 的
        # getJointInfo 里没有可用的父连杆索引）。
        parent_map = build_parent_map(tmp)
        print(f'  连杆父子关系：{len(parent_map)} 条（来自 URDF 本身）')

        # 指关节限位（拿真实限位，而不是硬编码）
        jl, jh = [], []
        for nm in ARM_JOINTS:
            lo, hi = joint_limits(pb, robot, nm)
            jl.append(lo)
            jh.append(hi)
        print('  关节限位（来自 URDF，实测应与 ±1.5708 一致）：')
        for nm, a, b in zip(ARM_JOINTS, jl, jh):
            print(f'    {nm:20s} [{a:+.4f}, {b:+.4f}]')

        rng = np.random.default_rng(20261003)

        # ---- ① 可达工作区 ----
        print(f'\n[①] 可达工作区：{args.samples} 个关节空间采样 ...')
        t0 = time.time()
        tcp_offset = tcp_offset_in_claw(tmp)
        print(f'  tcp 在 claw_base 局部系的偏置（从 URDF 现读）：'
              f'{np.round(tcp_offset, 4).tolist()} m')
        claw_idx = link_name_to_joint_index(pb, robot, 'claw_base')
        # 采样用的是**真实限位**而不是无界随机：否则会采到几何上不可能的
        # 位形，把工作区画得比实际大得多。
        lo_a = np.array(jl[:ARM_GROUP_END])
        hi_a = np.array(jh[:ARM_GROUP_END])
        arm_idx = [joint_name_to_index(pb, robot, n) for n in ARM_JOINTS]

        pts = np.zeros((args.samples, 3))
        filled = 0
        batch = 2000
        while filled < args.samples:
            k = min(batch, args.samples - filled)
            qs = rng.uniform(lo_a, hi_a, size=(k, len(arm_idx)))
            for i in range(k):
                for jj, qv in enumerate(qs[i]):
                    pb.resetJointState(robot, arm_idx[jj], float(qv))
                st = pb.getLinkState(robot, claw_idx,
                                     computeForwardKinematics=1)
                # getLinkState 的 worldLinkFrameOrientation 是**四元数**
                # (x,y,z,w)，不是旋转矩阵 —— reshape(3,3) 会报
                # "cannot reshape array of size 4"。用 getMatrixFromQuaternion
                # 转一下，别自己写四元数转矩阵（符号约定容易搞错）。
                R = np.array(pb.getMatrixFromQuaternion(st[5])).reshape(3, 3)
                pts[filled + i] = np.array(st[4]) + R @ tcp_offset
            filled += k
        print(f'  {args.samples} 个采样点用时 {time.time()-t0:.1f} s')
        print(f'  TCP 可达范围 x [{pts[:,0].min():+.4f}, {pts[:,0].max():+.4f}]'
              f'  y [{pts[:,1].min():+.4f}, {pts[:,1].max():+.4f}]'
              f'  z [{pts[:,2].min():+.4f}, {pts[:,2].max():+.4f}]')

        # ---- 网格 ----
        xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.grid)
        ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.grid)

        # 可达掩码：按 xy 分箱统计采样点密度，非空的格子算可达。
        # 为什么用直方图而不是包围盒：包围盒会把「可达但中间有大洞」的
        # 区域也算进去，而对抓取来说那个洞恰恰是不可用的。
        h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0],
                                 bins=[len(ys), len(xs)],
                                 range=[[ys[0] - args.grid / 2,
                                         ys[-1] + args.grid / 2],
                                        [xs[0] - args.grid / 2,
                                         xs[-1] + args.grid / 2]])
        reach = h > 0
        print(f'\n[网格] {len(xs)} x {len(ys)} = {reach.size} 格，'
              f'步长 {args.grid*1000:.0f} mm')
        print(f'  可达格数 {int(reach.sum())} / {reach.size}')

        # ---- ② 无碰撞 ----
        # lmap 在这里就要用（碰撞检测按连杆名判断相邻对），
        # 所以必须早于③ 之前建立 —— 之前放在 ③ 里是个顺序错误。
        lmap = link_index_map(pb, robot)
        if args.no_collision_check:
            collision_free = np.ones_like(reach)
            print('\n[②] 自碰撞检查：已跳过（--no-collision-check）')
        else:
            print('\n[②] 自碰撞检查：2000 个随机位形 ...')
            t0 = time.time()
            frac = _collision_free_frac(pb, robot, rng, parent_map, lmap)
            print(f'  无碰撞位形比例 {frac*100:.1f}%，用时 {time.time()-t0:.1f} s')
            collision_free = np.ones_like(reach)
            if frac < 0.5:
                collision_free[:] = False
                print('  → 无碰撞比例过低，全部标为不可用（检查模型是否正确）')

        # ---- ③ 无遮挡 ----
        print('\n[③] 手指遮挡分析（rayTestBatch，在扫描位形下）...')
        finger_ids = {n: lmap[n] for n in FINGER_LINKS if n in lmap}
        if not finger_ids:
            print('  **警告**：没找到手指连杆，遮挡分析会全部判为「无遮挡」。'
                  '这是一次「安静地给错答案」—— 检查 URDF 是否被改动。',
                  file=sys.stderr)
        print(f'  手指 rayTest linkIndex：'
              f'{ {n: lmap[n] for n in FINGER_LINKS if n in lmap} }')
        blocked, in_fov, cam_pos = fov_occlusion_map(pb, robot, xs, ys)
        print(f'  相机位置 ({cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, '
              f'{cam_pos[2]:+.4f})')
        print(f'  在视锥内 {int(in_fov.sum())} 格，被手指遮挡 '
              f'{int(blocked.sum())} 格')

        # ---- 交集 ----
        good = reach & collision_free & in_fov & ~blocked
        print(f'\n[交集] 可达 ∩ 无碰撞 ∩ 无遮挡 = {int(good.sum())} 格 '
              f'({100*good.sum()/good.size:.1f}%)')

        # ---- 输出 ----
        _print_ascii(xs, ys, reach, in_fov, blocked, good)
        if args.out:
            _save_png(args.out, xs, ys, reach, in_fov, blocked, good)
            print(f'\n热力图已保存：{args.out}')
        return 0
    finally:
        pb.disconnect(cid)


def _collision_free_frac(pb, robot, rng, parent_map, lmap, n=2000):
    """自碰撞位形比例（只在 5 个臂连杆之间检查）。

    只查臂连杆、不查手指：两根手指之间本来就该接触（夹爪闭合时
    夹住工件），把它们算成自碰撞会让所有闭合位形都被判无效。
    这个界定与实际任务一致 —— 我们要的是「摆位形不撞」。
    """
    ids = []
    for nm in ARM_JOINTS:
        lo, hi = joint_limits(pb, robot, nm)
        ids.append((joint_name_to_index(pb, robot, nm), lo, hi))
    body_links = [link_name_to_joint_index(pb, robot, n) for n in ARM_LINK_NAMES]
    bad = 0
    for _ in range(n):
        for i, lo, hi in ids:
            pb.resetJointState(robot, i, rng.uniform(lo, hi))
        hit = False
        for a in range(len(body_links)):
            for b in range(a + 1, len(body_links)):
                na, nb = ARM_LINK_NAMES[a], ARM_LINK_NAMES[b]
                if _linked(parent_map, na, nb):
                    continue          # 相邻对必然贴合，排除
                # 参数必须**只传 4 个**：本 API 的 getClosestPoints
                # 不接受 distance 关键字（实测 TypeError），而位置传
                # 距离也会报 "'float' object cannot be interpreted as an
                # integer" —— 这个版本的第 5 个参数是 linkIndexPair，
                # 不是距离阈值。不传就用默认阈值（0），正是我们要的：
                # 只关心「是否接触」。
                pts = pb.getClosestPoints(robot, robot,
                                          body_links[a], body_links[b])
                # getClosestPoints 返回元组：
                #   (contactDistance, positionA, positionB, normal,
                #    contactDistance, indexA, indexB, ...)
                # 实测 [8] 是 contactDistance。距离 0 视为接触 ——
                # 阈值取 1 mm 会漏掉「刚好擦上」的情况，而那正是
                # 关节限位附近最容易出问题的地方。
                if pts and pts[0][8] < 0.002:
                    hit = True
                    break
            if hit:
                break
        if hit:
            bad += 1
    return 1.0 - bad / max(n, 1)


def tcp_offset_in_claw(urdf_path):
    """算 tcp 在 claw_base 局部系里的偏置（从 URDF 现读）。

    为什么不写死常量：tcp 的固定关节 origin 写在 arm_core.xacro 里，
    那里一改，硬编码的常量就静默过期 —— 而偏置错了只会让可达工作区
    整体偏移几毫米，图看起来仍然「合理」。从 URDF 现读则不可能漂移。
    这与 grasp_geometry.py 的做法一致（它每次都重新展开 xacro，
    注释里明确说宁可多花一秒也不留缓存）。

    实现：把 tcp_joint 的 origin 与其各级父关节变换连乘到 claw_base。
    只做这一小段链（claw_base -> tcp 是 fixed 关节），所以直接读 origin
    就够，不需要完整 FK。
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    off = np.zeros(3)
    rot = np.eye(3)
    joint_by_child = {}
    for j in root.findall('joint'):
        child = j.find('child')
        if child is not None:
            joint_by_child[child.get('link')] = j
    link = TCP_LINK
    # 一路往上走到 claw_base，把每一级的 origin 变换累乘上去
    while link != 'claw_base':
        j = joint_by_child.get(link)
        if j is None:
            raise KeyError(f'连杆 {link} 不在 URDF 的关节树里（找不到它的父关节）')
        o = j.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz') or '0 0 0').split()]) \
            if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()] \
            if o is not None else [0, 0, 0]
        off = rot @ xyz + off
        rot = rot @ rpy_to_matrix(rpy)
        link = j.find('parent').get('link')
    return off


def build_parent_map(urdf_path):
    """从 URDF 解析出 连杆名 -> 父连杆名 的映射。

    为什么不用 PyBullet 的 getJointInfo 来推父子关系：
    实测（本API 返回 17 个元素）里**没有可用的父连杆索引**：
        [3]  是 childLinkIndex（该关节自己的 child，唯一）
        [16] 也是 childLinkIndex（同一批连杆上是 7..13）
    两者都不是 parent。我在这上面错了三轮，每一轮的后果都是
    「相邻连杆对没被排除」→ 自碰撞检测把**关节处本来就贴合**的
    相邻连杆报成碰撞 → 无碰撞比例接近 0 → 整张工作区图全黑，
    而程序不会报任何错。

    结论：父子关系只能从 URDF 自己读，那是唯一的权威来源。
    这样还有一个额外好处：相邻对的定义与 MoveIt 的 SRDF 语义一致
    （踩坑记录第 18 条也强调相邻连杆对必须显式禁用）。
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    parent = {}
    for j in root.findall('joint'):
        p = j.find('parent')
        c = j.find('child')
        if p is not None and c is not None:
            parent[c.get('link')] = p.get('link')
    return parent


def _linked(parent_map, li_name, lj_name):
    """两个连杆是否由关节直接相连（按名字判断，避免索引空间混用）。

    相邻连杆在关节处必然贴合，自碰撞检测必须排除它们 ——
    踩坑记录第 18 条：MoveIt 的碰撞矩阵里相邻连杆对不显式禁用就会
    报满屏碰撞。这里是同一个道理，只是发生在 PyBullet 里。
    """
    return (parent_map.get(li_name) == lj_name
            or parent_map.get(lj_name) == li_name)


def _print_ascii(xs, ys, reach, in_fov, blocked, good):
    """打印四张ASCII 图叠在一起看。

    为什么用字符而不是数字：数字需要对照表，字符扫一眼就知道。
    每个位置一个字符，优先级 遮挡 > 视野外 > 不可达 > 可用。
    """
    print('\n=== 台面网格（每格一个字符）===')
    print('  图例：# 手指遮挡   . 视野外/不可达   O 可达且无遮挡   * 可抓（交集）')
    print(f'  x: {xs[0]*1000:.0f} → {xs[-1]*1000:.0f} mm，'
          f'步长 {(xs[1]-xs[0])*1000:.0f} mm   '
          f'y: {ys[0]*1000:.0f} → {ys[-1]*1000:.0f} mm')
    for iy in range(len(ys) - 1, -1, -1):
        row = []
        for ix in range(len(xs)):
            if blocked[iy, ix]:
                row.append('#')
            elif not in_fov[iy, ix] or not reach[iy, ix]:
                row.append('.')
            elif good[iy, ix]:
                row.append('*')
            else:
                row.append('O')
        print(f'  y={ys[iy]*1000:3.0f}mm |' + ''.join(row) + '|')

    # 可用区间汇总
    print('\n=== 逐行可用区间（可直接抄进文档）===')
    for iy in range(len(ys) - 1, -1, -1):
        segs, start = [], None
        for ix in range(len(xs)):
            if good[iy, ix] and start is None:
                start = ix
            elif not good[iy, ix] and start is not None:
                segs.append((xs[start], xs[ix - 1]))
                start = None
        if start is not None:
            segs.append((xs[start], xs[-1]))
        if segs:
            txt = ', '.join(f'[{a*1000:.0f}, {b*1000:.0f}]' for a, b in segs)
            print(f'  y={ys[iy]*1000:3.0f} mm : x {txt} mm')
        else:
            print(f'  y={ys[iy]*1000:3.0f} mm : 无可用点')


def _save_png(path, xs, ys, reach, in_fov, blocked, good):
    """存热力图。matplotlib 来自 open3d 的依赖，已装。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('（无 matplotlib，跳过 PNG）')
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (data, title) in zip(axes, [
            (reach.astype(float), '① 可达工作区'),
            ((in_fov & ~blocked).astype(float), '③ 无遮挡（在视锥内且手指不挡）'),
            (good.astype(float), '交集：可抓')]):
        im = ax.imshow(data, origin='lower', extent=[xs[0], xs[-1],
                                                     ys[0], ys[-1]],
                       cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        # 标出方块的当前位置
        ax.plot(0.0267, 0.0245, 'r+', ms=12, mew=2, label='target_cube')
        ax.legend(loc='upper right', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  三联图：可达 / 无遮挡 / 交集，红十字是当前方块位置')


if __name__ == '__main__':
    sys.exit(main())