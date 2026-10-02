#!/usr/bin/env python3
"""diag_cluster.py — 诊断感知点簇的形状，并统计各中心估计量的**方差**

要回答两个问题：
  1. xy 误差的不对称来自哪里？  → 看点簇在 x / y 上的分布，以及手指碰撞盒
     相对点簇的位置（判断是不是自遮挡截断）
  2. 哪种中心估计量最稳？      → 连续采 N 帧，比较包围盒中心 / 质心 / 中位数 /
     分位数区间的**误差与标准差**。单次看不出方差，必须多次采。

用法：
    /usr/bin/python3 ~/arm5_ws/tools/diag_cluster.py            # 形状诊断
    /usr/bin/python3 ~/arm5_ws/tools/diag_cluster.py --repeat 8 # 采 8 帧统计方差
"""
import argparse
import os
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
                                  cloud_to_xyz, quat_to_matrix, segment, self_filter,
                                  gazebo_truth)
from grasp_geometry import fk as arm_fk, BOX as CLAW_BOX

CUBE = 'target_cube'
TRUE_SIZE = 0.025

# 扫描位形（SRDF grasp_ready）。诊断前必须核对臂在这个位形上，否则测出来的
# 数字没有意义 —— 这是本脚本自己踩过的坑：曾在一串"臂停在撤离位/限位附近"的
# 测量上得出结论，跨度从 26.8 变 19.3，误以为是物体位置导致的。
SCAN_POSE = {'top_plate_joint': 0.0, 'lower_arm_joint': -0.0385,
             'upper_arm_joint': -0.1531, 'wrist_joint': -1.45,
             'claw_base_joint': 0.0}
SCAN_TOL = 0.02       # rad


def report_arm_pose(joints_msg):
    """打印当前关节角，并判断是否处于扫描位形。返回是否匹配。"""
    cur = dict(zip(joints_msg.name, joints_msg.position))
    parts, worst, worst_name = [], 0.0, ''
    for j, want in SCAN_POSE.items():
        got = cur.get(j)
        if got is None:
            parts.append(f'{j}=缺失')
            continue
        d = abs(got - want)
        parts.append(f'{j.replace("_joint","")[:5]}={got:+.3f}')
        if d > worst:
            worst, worst_name = d, j
    ok = worst <= SCAN_TOL
    print(f'  臂位形 [{" ".join(parts)}]  '
          f'{"✓ 在扫描位形" if ok else f"✗ 偏离 {worst_name} {worst:.3f} rad"}')
    if not ok:
        print('    ⚠ 这组数据不可与其它次比较 —— 视角变了，可见面积就变了。'
              '请先把臂摆到 grasp_ready 再测。')
    return ok

# 各种中心估计量。名字会直接打进统计表。
ESTIMATORS = {
    'bbox':      lambda o: (0.5 * (o[:, 0].min() + o[:, 0].max()),
                            0.5 * (o[:, 1].min() + o[:, 1].max())),
    'centroid':  lambda o: (float(o[:, 0].mean()), float(o[:, 1].mean())),
    'median':    lambda o: (float(np.median(o[:, 0])), float(np.median(o[:, 1]))),
    'p1p99':     lambda o: (0.5 * (np.percentile(o[:, 0], 1) + np.percentile(o[:, 0], 99)),
                            0.5 * (np.percentile(o[:, 1], 1) + np.percentile(o[:, 1], 99))),
    'p2p98':     lambda o: (0.5 * (np.percentile(o[:, 0], 2) + np.percentile(o[:, 0], 98)),
                            0.5 * (np.percentile(o[:, 1], 2) + np.percentile(o[:, 1], 98))),
    'p5p95':     lambda o: (0.5 * (np.percentile(o[:, 0], 5) + np.percentile(o[:, 0], 95)),
                            0.5 * (np.percentile(o[:, 1], 5) + np.percentile(o[:, 1], 95))),
}


def histogram(vals, lo, hi, nbin=20):
    edges = np.linspace(lo, hi, nbin + 1)
    cnt, _ = np.histogram(vals, bins=edges)
    mx = max(cnt.max(), 1)
    for i, c in enumerate(cnt):
        print(f'    {edges[i]:+.4f}..{edges[i+1]:+.4f} '
              f'{"#" * int(c / mx * 46):<46} {c}')


def capture(node, tfb, cloud, joints):
    """采一帧，返回 (点簇, 地面高度) 或 (None, None)。"""
    cloud.clear()
    joints.clear()
    t0 = time.time()
    while ('m' not in cloud or 'm' not in joints) and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'm' not in cloud:
        return None, None
    xyz = cloud_to_xyz(cloud['m'])
    try:
        tr = tfb.lookup_transform('base_link', POINTS_DATA_FRAME, Time(),
                                  timeout=Duration(seconds=3.0))
    except Exception as exc:                          # noqa: BLE001
        print(f'  TF 查询失败：{exc}', file=sys.stderr)
        return None, None
    t, q = tr.transform.translation, tr.transform.rotation
    R = quat_to_matrix(q.x, q.y, q.z, q.w)
    xyz = xyz @ R.T + np.array([t.x, t.y, t.z])

    jm = joints['m']
    xyz = xyz[self_filter(xyz, arm_fk(dict(zip(jm.name, jm.position))), 0.004)]
    xyz = xyz[xyz[:, 2] <= 0.035]
    clusters, z_ground, _ = segment(xyz, 0.004, 0.002, 30)
    if not clusters:
        return None, z_ground
    return clusters[0], z_ground


def main():
    ap = argparse.ArgumentParser(description='诊断感知点簇形状与中心估计方差')
    ap.add_argument('--repeat', type=int, default=0,
                    help='连续采 N 帧，统计各估计量的误差与标准差')
    ap.add_argument('--settle', type=float, default=2.0,
                    help='每帧之间的等待秒数（相机 10Hz）')
    ap.add_argument('--force', action='store_true',
                    help='即使臂不在扫描位形也强行测量（结果不可比）')
    args = ap.parse_args()

    rclpy.init()
    node = Node('diag_cluster',
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

    truth = gazebo_truth(CUBE)
    print(f'Gazebo 真值 {truth}\n')

    # ---------- 方差统计模式 ----------
    if args.repeat:
        errs = {k: [] for k in ESTIMATORS}
        npts, spans = [], []
        # 采样前先核对臂位形并记下整段的实际位形 —— 诊断过程中臂要是被别的
        # 动作带动（流水线执行完会停在撤离位），这一段数据就废了。
        t_wait = time.time()
        while 'm' not in joints and time.time() - t_wait < 10.0:
            rclpy.spin_once(node, timeout_sec=0.2)
        if 'm' not in joints:
            print('收不到 /joint_states，无法核对臂位形')
            node.destroy_node()
            rclpy.shutdown()
            return 6
        arm_ok = report_arm_pose(joints['m'])
        if not arm_ok and not args.force:
            print('\n臂不在扫描位形，结果不可比。要强行测量请加 --force。')
            node.destroy_node()
            rclpy.shutdown()
            return 5
        for i in range(args.repeat):
            obj, _ = capture(node, tfb, cloud, joints)
            if obj is None:
                print(f'  第 {i+1} 帧没聚出簇')
                continue
            tr = gazebo_truth(CUBE)
            npts.append(len(obj))
            spans.append((obj[:, 0].max() - obj[:, 0].min(),
                          obj[:, 1].max() - obj[:, 1].min()))
            for k, fn in ESTIMATORS.items():
                cx, cy = fn(obj)
                errs[k].append(((cx - tr[0]) * 1000, (cy - tr[1]) * 1000))
            print(f'  第 {i+1} 帧 {len(obj):5d} 点  '
                  f'x 跨度 {(spans[-1][0])*1000:5.1f} mm  '
                  f'y 跨度 {(spans[-1][1])*1000:5.1f} mm')
            time.sleep(args.settle)

        print(f'\n=== 各估计量的误差统计（{len(npts)} 帧有效）===')
        print('  点数范围 %d ~ %d    x 跨度均值 %.1f mm    y 跨度均值 %.1f mm'
              % (min(npts), max(npts),
                 np.mean([s[0] for s in spans]) * 1000,
                 np.mean([s[1] for s in spans]) * 1000))
        print('  真值宽度 25.0 mm\n')
        print('  估计量        x 均值   x 标准差   x 最差     y 均值   y 标准差   y 最差')
        for k, v in errs.items():
            if not v:
                continue
            a = np.array(v)
            print(f'  {k:10s} {a[:,0].mean():+7.2f}  {a[:,0].std():8.2f}  '
                  f'{np.abs(a[:,0]).max():6.2f}   '
                  f'{a[:,1].mean():+7.2f}  {a[:,1].std():8.2f}  '
                  f'{np.abs(a[:,1]).max():6.2f}    (mm)')
        node.destroy_node()
        rclpy.shutdown()
        return 0

    # ---------- 形状诊断模式 ----------
    time.sleep(0.5)
    rclpy.spin_once(node, timeout_sec=0.3)
    report_arm_pose(joints['m'])
    obj, z_ground = capture(node, tfb, cloud, joints)
    if obj is None:
        print('没聚出簇', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 4

    lo, hi = obj.min(axis=0), obj.max(axis=0)
    print(f'点簇 {len(obj)} 点    地面高度 z = {z_ground:+.4f}')
    print(f'  x 跨度 {(hi[0]-lo[0])*1000:5.1f} mm   （真值 25.0）')
    print(f'  y 跨度 {(hi[1]-lo[1])*1000:5.1f} mm')
    print(f'  z 区间 [{lo[2]:+.4f}, {hi[2]:+.4f}]')

    print('\n=== x 方向分布 ===')
    histogram(obj[:, 0], lo[0], hi[0])
    print('\n=== y 方向分布（注意第二瓣）===')
    histogram(obj[:, 1], lo[1], hi[1])

    print('\n=== 各中心估计量 vs 真值 ===')
    for k, fn in ESTIMATORS.items():
        cx, cy = fn(obj)
        print(f'  {k:10s} x {cx:+.4f} (Δ{(cx-truth[0])*1000:+6.2f} mm)   '
              f'y {cy:+.4f} (Δ{(cy-truth[1])*1000:+6.2f} mm)')

    print('\n=== 两指碰撞盒 vs 点簇（判断自遮挡是否成立）===')
    fk_res = arm_fk(dict(zip(joints['m'].name, joints['m'].position)))
    for link, (org, size) in CLAW_BOX.items():
        Rr, tt = fk_res[link]
        Rm, tm = np.asarray(Rr, float), np.asarray(tt, float)
        corners = np.array([[sx*size[0]/2, sy*size[1]/2, sz*size[2]/2]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        w = corners @ Rm.T + tm + np.asarray(org, float)
        print(f'  {link:12s} x [{w[:,0].min():+.4f}, {w[:,0].max():+.4f}]  '
              f'z [{w[:,2].min():+.4f}, {w[:,2].max():+.4f}]')
    print(f'  {"点簇":12s} x [{lo[0]:+.4f}, {hi[0]:+.4f}]  z [{lo[2]:+.4f}, {hi[2]:+.4f}]')
    print(f'  {"真值方块":10s} x [{truth[0]-TRUE_SIZE/2:+.4f}, {truth[0]+TRUE_SIZE/2:+.4f}]'
          f'  z [{truth[2]-TRUE_SIZE/2:+.4f}, {truth[2]+TRUE_SIZE/2:+.4f}]')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
