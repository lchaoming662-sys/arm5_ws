#!/usr/bin/env python3
"""ik_compare_bench.py — KDL vs Pick-IK 的 /compute_ik 纯 IK 对比测试

背景（docs/踩坑记录.md 第 17/31 条 + Task 4）：
    KDL 在 position_only_ik 下会把冗余自由度推到硬限位上（wrist/claw_base
    实测 -1.57080 = 限位本身）。Pick-IK 有内置的 avoid_joint_limits_weight
    代价函数，理论上能把解推离限位。本脚本用同一批目标、同一把种子，
    对两个求解器做受控对比。

测量前置条件（本脚本启动时逐条自检，任一不过直接退出非 0）：
    1. /compute_ik 服务在线（move_group 活着）
    2. /move_group 的 kinematics_solver 参数 = 本次测试想要测的那个求解器
       —— 这是「写了 vs 生效了」的闸门：参数没加载时 MoveIt 不报错、
       照旧用 KDL，必须单独查（见 kinematics.yaml 头部注释）
    3. move_group 收得到 /joint_states（本脚本自己以 10 Hz 发布 home 零位，
       所以纯 IK 测试不需要 Gazebo；计时走墙上时钟，不受 RTF 干扰）

公平性控制：
    · 同一批 20 个目标位姿（确定性生成，两次运行逐个相同）
    · 同一把种子（home 全零，随请求显式下发，不依赖当前状态）
    · 同一请求超时 0.5 s
    · 成功判据：error_code.val == 1（SUCCESS）

输出：
    · 逐目标表格（stdout）
    · 汇总指标：成功率 / 求解时间中位数与均值 / wrist 与 claw_base
      距硬限位 ±1.5708 的最小余量（rad）
    · JSON 结果文件（--out 指定），供跨求解器汇总对比

用法：
    python3 tools/ik_compare_bench.py --label kdl --out /tmp/ik_bench_kdl.json
    python3 tools/ik_compare_bench.py --label pickik_baseline --out /tmp/ik_bench_p0.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState as RobotStateMsg, PositionIKRequest
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState as SeedJointState

# ---------------------------------------------------------------------------
# 常量：与工程其他地方保持一致（见 tools/joint_limit_cost.py 的 HARD_LIMITS）
# ---------------------------------------------------------------------------
HARD_LIMIT = 1.5708          # URDF 物理硬限位 ±1.5708（arm_core.xacro）
ARM_JOINTS = [
    'top_plate_joint',
    'lower_arm_joint',
    'upper_arm_joint',
    'wrist_joint',
    'claw_base_joint',
]
KEY_JOINTS = ['wrist_joint', 'claw_base_joint']   # 实测会被 KDL 顶到限位的两个
IK_LINK = 'tcp'
GROUP = 'arm'
TIMEOUT_S = 0.5               # 两个求解器用同一个请求超时，保证公平

# 目标生成：x 固定在抓取工作区（踩坑记录第 17 条验证命令用的 0.0267），
# y/z 网格 5×4 = 20 个，确定性（不随机，两次运行必须逐个相同）
TARGET_X = 0.0267
Y_GRID = [-0.16, -0.08, 0.0, 0.08, 0.16]
Z_GRID = [0.14, 0.1933, 0.2467, 0.30]


def make_targets():
    targets = []
    for z in Z_GRID:
        for y in Y_GRID:
            targets.append({
                'x': TARGET_X, 'y': y, 'z': z,
                'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
            })
    return targets


class IKBench(Node):
    def __init__(self):
        super().__init__('ik_compare_bench')
        # 自己发 /joint_states：move_group 的 CurrentStateMonitor 靠它确认
        # 「当前状态」可用。纯 IK 测试不需要 Gazebo，零位即可。
        self._js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self._js_timer = self.create_timer(0.1, self._publish_joint_states)
        self._client = self.create_client(GetPositionIK, '/compute_ik')

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINTS + ['right_claw_joint']
        msg.position = [0.0] * len(msg.name)
        self._js_pub.publish(msg)

    # ---------------- 前置条件自检 ----------------
    def wait_service(self, timeout_s=10.0):
        if not self._client.wait_for_service(timeout_sec=timeout_s):
            print('[自检失败] /compute_ik 服务不在线 —— move_group 没起来？')
            sys.exit(2)
        print('[自检通过] /compute_ik 服务在线')

    def check_solver(self, expect_substring: str | None):
        """查 /move_group 实际加载的求解器参数。

        这一步必须做：kinematics.yaml 写了什么不等于加载了什么。
        参数没加载时 MoveIt 静默退回 KDL，不报任何错。
        expect_substring 为 None 表示只打印不判定。
        """
        proc = subprocess.run(
            ['ros2', 'param', 'get', '/move_group',
             'robot_description_kinematics.arm.kinematics_solver'],
            capture_output=True, text=True, timeout=15)
        out = (proc.stdout or '').strip()
        print(f'[自检] move_group 实际加载的求解器: {out!r}')
        if expect_substring and expect_substring not in out:
            print(f'[自检失败] 期望包含 {expect_substring!r}，实际如上'
                  ' —— 配置写了但没被加载，禁止出数据')
            sys.exit(3)

    def warmup(self):
        """等 CurrentStateMonitor 吃到我们发的 /joint_states。

        判据：第一次 IK 调用成功或返回明确 error_code（而不是异常/超时）。
        """
        for attempt in range(10):
            code, _, _ = self.solve_one(make_targets()[0], quiet=True)
            if code is not None:
                print(f'[自检通过] move_group 已收到机器人状态（warmup 第'
                      f' {attempt + 1} 次，error_code={code}）')
                return
            time.sleep(0.5)
        print('[自检失败] move_group 迟迟拿不到 /joint_states，测量不可信')
        sys.exit(4)

    # ---------------- 单目标求解 ----------------
    def solve_one(self, tgt, quiet=False):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = IK_LINK
        req.ik_request.timeout = Duration(seconds=TIMEOUT_S).to_msg()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped.header.frame_id = 'world'
        req.ik_request.pose_stamped.pose = Pose(
            position=Point(x=tgt['x'], y=tgt['y'], z=tgt['z']),
            orientation=Quaternion(w=1.0),
        )
        # 显式下发种子（home 全零）：KDL 从种子出发做数值迭代，
        # Pick-IK global 虽不依赖初值，但两边喂同一把种子才叫受控对比
        seed = JointState()
        seed.name = ARM_JOINTS + ['right_claw_joint']
        seed.position = [0.0] * len(seed.name)
        req.ik_request.robot_state = RobotStateMsg(joint_state=seed)

        t0 = time.perf_counter()
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        wall_s = time.perf_counter() - t0

        if future.result() is None:
            print('[异常] /compute_ik 调用超时/无响应 —— 测量本身不可信')
            return None, None, wall_s
        resp = future.result()
        err = resp.error_code.val
        sol = {}
        if err == 1 and resp.solution.joint_state.name:
            sol = dict(zip(resp.solution.joint_state.name,
                           resp.solution.joint_state.position))
        if not quiet:
            if err == 1:
                margin_str = ', '.join(
                    f"{j.replace('_joint', '')}={HARD_LIMIT - abs(sol[j]):.4f}rad"
                    for j in KEY_JOINTS if j in sol)
            else:
                margin_str = '—'
            print(f"  y={tgt['y']:+.3f} z={tgt['z']:.3f}  "
                  f"err={err:+d}  t={wall_s * 1000:7.1f}ms  {margin_str}")
        return err, sol, wall_s

    # ---------------- 整批测试 ----------------
    def run_batch(self, label: str):
        targets = make_targets()
        print(f'\n===== 批次 {label}：{len(targets)} 个目标，'
              f'超时 {TIMEOUT_S}s，种子 home 全零 =====')
        rows = []
        for i, tgt in enumerate(targets):
            err, sol, wall_s = self.solve_one(tgt)
            row = {'idx': i, 'target': tgt, 'error_code': err,
                   'wall_s': wall_s, 'solution': sol}
            rows.append(row)

        ok = [r for r in rows if r['error_code'] == 1]
        times_ok = [r['wall_s'] for r in ok]
        summary = {
            'label': label,
            'n_targets': len(targets),
            'n_success': len(ok),
            'success_rate': len(ok) / len(targets),
            'time_median_ms': statistics.median(times_ok) * 1000 if ok else None,
            'time_mean_ms': statistics.fmean(times_ok) * 1000 if ok else None,
            'time_max_ms': max(times_ok) * 1000 if ok else None,
        }
        # 距硬限位余量：只对成功解统计（失败没有解可言）
        for j in KEY_JOINTS:
            margins = [HARD_LIMIT - abs(r['solution'][j]) for r in ok if j in r['solution']]
            summary[f'{j}_min_margin_rad'] = min(margins) if margins else None
            summary[f'{j}_median_margin_rad'] = statistics.median(margins) if margins else None
        # 全部关节里最小的余量（看有没有别的关节被推到边界）
        all_margins = []
        for r in ok:
            for j in ARM_JOINTS:
                if j in r['solution']:
                    all_margins.append(HARD_LIMIT - abs(r['solution'][j]))
        summary['any_joint_min_margin_rad'] = min(all_margins) if all_margins else None
        summary['rows'] = rows
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default=None, help='JSON 结果文件路径')
    ap.add_argument('--expect-solver', default=None,
                    help='求解器参数里必须包含的子串（写了 vs 生效了的闸门）')
    args = ap.parse_args()

    rclpy.init()
    node = IKBench()
    node.wait_service()
    node.check_solver(args.expect_solver)
    node.warmup()

    summary = node.run_batch(args.label)

    print('\n===== 汇总 =====')
    print(f"成功率: {summary['n_success']}/{summary['n_targets']}")
    if summary['time_median_ms'] is not None:
        print(f"求解时间: 中位 {summary['time_median_ms']:.1f} ms, "
              f"均值 {summary['time_mean_ms']:.1f} ms, "
              f"最坏 {summary['time_max_ms']:.1f} ms")
        for j in KEY_JOINTS:
            mn = summary[f'{j}_min_margin_rad']
            md = summary[f'{j}_median_margin_rad']
            print(f"{j}: 距硬限位余量 最小 {mn:.4f} rad（{mn * 180 / 3.14159:.1f}°），"
                  f"中位 {md:.4f} rad")
        a = summary['any_joint_min_margin_rad']
        print(f"全部关节最小余量: {a:.4f} rad")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f'结果已写入 {args.out}')

    node.destroy_node()
    rclpy.shutdown()
    # 有失败的解时退出码 1，方便脚本串联判断
    sys.exit(0 if summary['n_success'] == summary['n_targets'] else 1)


if __name__ == '__main__':
    main()
