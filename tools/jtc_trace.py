#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jtc_trace.py — 采样 arm_controller 的 reference / feedback / error，诊断容差违规

为什么用 /arm_controller/state 而不是自己算误差：
    它是一个 JointTrajectoryControllerState，里面 **控制器自己** 填好了
    reference（它当时认为该到的位置）和 error（reference − feedback）。
    自己拿 /joint_states 算，得先重建参考轨迹，反而引入了新的误差来源。
    容差检查本来就是控制器拿这两个数做的，所以直接采它最忠实。

用法：
    jtc_trace.py <输出csv> [采样秒数]

采样期间另开一个终端发目标（见 docs/使用说明书.md 5.7 的 /move_action 例子）。
CSV 每行：t, ref_*, fb_*, err_*, act_*（act = /joint_states 的真实位置；
本机 ref_* 与 fb_* 是空的，所以用 ref ≈ act + err 反推）
"""

import csv
import sys
import time

import rclpy
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

JOINTS = ['top_plate_joint', 'lower_arm_joint', 'upper_arm_joint',
          'wrist_joint', 'claw_base_joint']


def _pad(values, n):
    out = list(values[:n])
    out += [float('nan')] * (n - len(out))
    return out


class Tracer(Node):

    def __init__(self, path):
        super().__init__('jtc_trace')
        self._fh = open(path, 'w', newline='')
        self._w = csv.writer(self._fh)
        self._w.writerow(
            ['t']
            + [f'ref_{j}' for j in JOINTS]
            + [f'fb_{j}' for j in JOINTS]
            + [f'err_{j}' for j in JOINTS]
            + [f'act_{j}' for j in JOINTS])
        self.samples = 0
        # 控制器顺序可能与 URDF 顺序不同，所以按名字重新排一遍
        self._order = None
        self._actual = {}          # 最近一帧 /joint_states 的真实位置（按名字）
        self.create_subscription(
            JointTrajectoryControllerState, '/arm_controller/state', self._cb, 100)
        # 控制器自报的 error = reference − 真实位置，但 reference 字段
        # 在本机是空的；所以再单独采一路 /joint_states 拿到真实位置，
        # 就能反推 reference = err + 真实位置。
        self.create_subscription(JointState, '/joint_states', self._cb_js, 100)

    def _cb_js(self, msg):
        self._actual = dict(zip(msg.name, msg.position))

    def _idx(self, names):
        if self._order is None:
            self._order = [names.index(j) if j in names else None for j in JOINTS]
        return self._order

    def _cb(self, msg):
        names = list(msg.joint_names)
        if not names:
            return
        idx = self._idx(names)
        n = len(names)

        def pick(vals):
            vals = _pad(vals, n)
            return [float('nan') if i is None else vals[i] for i in idx]

        ref = pick(msg.reference.positions)
        # feedback 可能带 velocity，只取前 n 个当作位置
        fb = pick(msg.feedback.positions)
        if all(v != v for v in fb):          # 全是 nan → 这一版没有 feedback
            fb = [float('nan')] * len(JOINTS)
        err = pick(msg.error.positions)
        # /joint_states 的真实位置（可能比 state 帧稍慢一点点，100 Hz 同速）
        act = [self._actual.get(j, float('nan')) for j in JOINTS]

        self._w.writerow([f'{time.time():.4f}']
                         + [f'{v:.5f}' for v in ref]
                         + [f'{v:.5f}' for v in fb]
                         + [f'{v:.5f}' for v in err]
                         + [f'{v:.5f}' for v in act])
        self.samples += 1

    def close(self):
        self._fh.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0

    rclpy.init()
    node = Tracer(path)
    t0 = time.time()
    while time.time() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.close()
    print(f'采样结束：{node.samples} 条 -> {path}')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
