#!/usr/bin/env python3
"""map_workspace.py — 画出感知有效工作区地图

回答一个运维必须知道的问题：**目标放在台面哪里，这套系统能可靠感知到它？**

为什么必须画这张图：腕部相机视角里**自己的两指也在**（两指之间净空
29.3 mm，见 docs/使用说明书.md）。手指在桌面上投出两条遮挡带，目标落在
带子里就只看得见一块碎片，中心算出来能偏 10 mm（见 docs/踩坑记录.md
第 42 条）。而视场本身有 122 × 92 mm，远大于 25 mm 的方块 —— 所以
**限制不是视场不够大，是手指挡视线**。

判据与 estimate_object_pose.py 一致：实测跨度 ∈ [0.8, 1.8] × 工件尺寸。

用法：
    # 先把臂摆到扫描位形
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace.py
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace.py --grid 0.005
    /usr/bin/python3 ~/arm5_ws/tools/map_workspace.py --x-range 0.0 0.06 --y-range 0.0 0.05

地图以 ASCII 输出，`.` = 可用，`#` = 被遮挡/不可用，`?` = 没聚出簇。
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
import tf2_ros

from estimate_object_pose import (POINTS_TOPIC, JOINTS_TOPIC, POINTS_DATA_FRAME,
                                  cloud_to_xyz, quat_to_matrix, segment, self_filter)
from grasp_geometry import fk as arm_fk, BOX as CLAW_BOX

CUBE = 'target_cube'
WORLD = 'arm_world'
EXPECT_SIZE = 0.025
MIN_SPAN_RATIO, MAX_SPAN_RATIO = 0.80, 1.80
MIN_POINTS = 400

SCAN_POSE = {'top_plate_joint': 0.0, 'lower_arm_joint': -0.0385,
             'upper_arm_joint': -0.1531, 'wrist_joint': -1.45,
             'claw_base_joint': 0.0}
SCAN_TOL = 0.02

GROUND_TOL = 0.004
VOXEL = 0.002
MIN_CLUSTER_PTS = 30
Z_MAX = 0.035
SELF_MARGIN = 0.004


def teleport_cube(x, y, z=0.0125):
    """把方块瞬移到 (x, y)。写入必须走 set_pose 服务，ign model -p 是只读的。"""
    req = (f'ign service -s /world/{WORLD}/set_pose '
           f'--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
           f'--timeout 5000 --req '
           f'\'name: "{CUBE}" position: {{x: {x}, y: {y}, z: {z}}} '
           f'orientation: {{w: 1.0}}\'')
    return subprocess.run(['bash', '-c', req], capture_output=True,
                          text=True, timeout=40).returncode == 0


def cube_pose():
    r = subprocess.run(['ign', 'model', '-m', CUBE, '-p'],
                       capture_output=True, text=True, timeout=15)
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                v = [float(x) for x in s[1:-1].split()]
            except ValueError:
                continue
            if len(v) == 3:
                return v
    return None


def arm_pose(joints_msg):
    cur = dict(zip(joints_msg.name, joints_msg.position))
    worst, name = 0.0, ''
    for j, want in SCAN_POSE.items():
        if j in cur:
            d = abs(cur[j] - want)
            if d > worst:
                worst, name = d, j
    return worst <= SCAN_TOL, worst, name


def main():
    ap = argparse.ArgumentParser(description='画出感知有效工作区地图')
    ap.add_argument('--x-range', type=float, nargs=2, default=[0.000, 0.055],
                    metavar=('XMIN', 'XMAX'))
    ap.add_argument('--y-range', type=float, nargs=2, default=[0.005, 0.050],
                    metavar=('YMIN', 'YMAX'))
    ap.add_argument('--grid', type=float, default=0.005, help='网格步长（米）')
    ap.add_argument('--settle', type=float, default=1.6,
                    help='每次移动后方块落定 + 相机出帧的等待秒数')
    ap.add_argument('--repeat', type=int, default=1,
                    help='每格采几帧取中位（默认 1）')
    args = ap.parse_args()

    rclpy.init()
    node = Node('map_workspace',
                parameter_overrides=[Parameter('use_sim_time', value=True)])
    cloud, joints = {}, {}
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(PointCloud2, POINTS_TOPIC,
                             lambda m: cloud.__setitem__('m', m), qos)
    node.create_subscription(JointState, JOINTS_TOPIC,
                             lambda m: joints.__setitem__('m', m), qos)
    tfb = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfb, node)

    t0 = time.time()
    while 'm' not in joints and time.time() - t0 < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'm' not in joints:
        print('收不到 /joint_states', file=sys.stderr)
        return 2
    ok, worst, name = arm_pose(joints['m'])
    if not ok:
        print(f'臂不在扫描位形（{name} 偏 {worst:.3f} rad）。'
              f'请先摆到 grasp_ready 再画。', file=sys.stderr)
        return 3
    print(f'臂在扫描位形 ✓  工件 {EXPECT_SIZE*1000:.0f} mm  '
          f'判据 跨度 ∈ [{MIN_SPAN_RATIO*EXPECT_SIZE*1000:.0f}, '
          f'{MAX_SPAN_RATIO*EXPECT_SIZE*1000:.0f}] mm  点数 ≥ {MIN_POINTS}')

    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.grid)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.grid)

    # 两指在 base_link 下的 x 区间 —— 用来验证「失败格 = 手指遮挡带」
    fk_res = arm_fk(dict(zip(joints['m'].name, joints['m'].position)))
    finger_x = []
    for link in CLAW_BOX:
        Rr, tt = fk_res[link]
        Rm, tm = np.asarray(Rr, float), np.asarray(tt, float)
        org, size = CLAW_BOX[link]
        corners = np.array([[sx*size[0]/2, sy*size[1]/2, sz*size[2]/2]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        w = corners @ Rm.T + tm + np.asarray(org, float)
        finger_x.append((link, w[:, 0].min(), w[:, 0].max()))
        print(f'  {link:12s} x [{w[:,0].min():+.4f}, {w[:,0].max():+.4f}]')
    print()

    grid = {}
    for yi, y in enumerate(ys):
        row = []
        for xi, x in enumerate(xs):
            teleport_cube(float(x), float(y))
            time.sleep(args.settle)
            verdict, span, npts = '.', 0.0, 0
            for _ in range(args.repeat):
                cloud.clear()
                tw = time.time()
                while 'm' not in cloud and time.time() - tw < 8.0:
                    rclpy.spin_once(node, timeout_sec=0.1)
                if 'm' not in cloud:
                    continue
                xyz = cloud_to_xyz(cloud['m'])
                try:
                    tr = tfb.lookup_transform('base_link', POINTS_DATA_FRAME, Time(),
                                              timeout=Duration(seconds=2.0))
                except Exception:                      # noqa: BLE001
                    continue
                tt_, q = tr.transform.translation, tr.transform.rotation
                R = quat_to_matrix(q.x, q.y, q.z, q.w)
                xyz = xyz @ R.T + np.array([tt_.x, tt_.y, tt_.z])
                jm = joints['m']
                xyz = xyz[self_filter(xyz, arm_fk(dict(zip(jm.name, jm.position))),
                                      SELF_MARGIN)]
                xyz = xyz[xyz[:, 2] <= Z_MAX]
                clusters, _, _ = segment(xyz, GROUND_TOL, VOXEL, MIN_CLUSTER_PTS,
                                         verbose=False)
                if not clusters:
                    verdict = '?'
                    continue
                o = clusters[0]
                sx = float(np.percentile(o[:, 0], 99) - np.percentile(o[:, 0], 1))
                sy = float(np.percentile(o[:, 1], 99) - np.percentile(o[:, 1], 1))
                span, npts = min(sx, sy), len(o)
                lo_lim = EXPECT_SIZE * MIN_SPAN_RATIO
                hi_lim = EXPECT_SIZE * MAX_SPAN_RATIO
                good = all(lo_lim <= s <= hi_lim for s in (sx, sy)) \
                    and len(o) >= MIN_POINTS
                verdict = '.' if good else '#'
            grid[(xi, yi)] = (verdict, span * 1000, npts)
            row.append(verdict)
        print(f'  y={y:+.3f}  ' + ' '.join(row))

    print(f'\n图例  . 可用   # 跨度/点数不合格   ? 没聚出簇')
    print(f'      x 从 {xs[0]:+.3f} 到 {xs[-1]:+.3f}，步长 {args.grid*1000:.0f} mm')
    print(f'      y 从 {ys[0]:+.3f} 到 {ys[-1]:+.3f}')

    # 汇总可用区间（沿 x 与沿 y 各一行，便于直接抄进文档）
    print('\n=== 可用区间（逐行 y 统计 x 方向连续可用段）===')
    for yi, y in enumerate(ys):
        segs, start = [], None
        for xi in range(len(xs)):
            v = grid[(xi, yi)][0]
            if v == '.' and start is None:
                start = xi
            elif v != '.' and start is not None:
                segs.append((xs[start], xs[xi - 1]))
                start = None
        if start is not None:
            segs.append((xs[start], xs[-1]))
        if segs:
            txt = ', '.join(f'[{a*1000:.0f}, {b*1000:.0f}]' for a, b in segs)
            print(f'  y={y*1000:3.0f} mm : x {txt} mm')
        else:
            print(f'  y={y*1000:3.0f} mm : 无可用点')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
