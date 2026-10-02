#!/usr/bin/env python3
"""pick_demo.py — O5-2 机械臂「抓取闭环」演示

把序列走完一遍并给出判定：

    张开 -> 预抓取 -> 下压到抓取位 -> 闭合 -> 抬起 -> 判定方块是否被提起
         -> 放回 -> 张开 -> 回零位

判定依据不靠 ROS 层的关节值，而是**直接问 Gazebo 方块的真实位姿**
（`ign model -m target_cube -p`），所以「夹爪闭合了但方块滑掉了」这种
情况会被如实报出来。

用法（先 source 本工作区，仿真要在跑）：
    python3 tools/pick_demo.py
"""

import re
import subprocess
import sys
import time

SRC = ('source /opt/ros/humble/setup.bash && '
       'source ~/arm5_ws/install/setup.bash && ')

JOINTS = ['top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 'wrist_joint', 'claw_base_joint']

# 关节位形：top_plate_joint=claw_base_joint=0，只动 lower_arm_joint/3/4。
# wrist_joint 原解为 -1.5708（正好压在限位上），这里退到 -1.55 留一点余量。
HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
PREGRASP = [0.0, -0.0385, -0.1531, -1.5500, 0.0]
GRASP = [0.0, -0.3698, -0.3133, -1.5142, 0.0]
LIFT = [0.0, +0.7229, -0.3174, -1.5500, 0.0]

GRIP_OPEN = 0.0        # 两指竖直张开位（夹持面间距 29.28 mm）
GRIP_CLOSE = -0.20     # 25 mm 方块在 -0.1465 rad 处刚好被贴住；留 ~1.7 mm 余量
                       # （-0.35 会多压 6.9 mm，方块刚体压不进去，会被挤跑）
                       # 详细的几何表见 tools/pick_demo_moveit.py
GRIP_EFFORT = 20.0

CUBE_LIFT_THRESHOLD = 0.05   # 方块中心抬升超过 5 cm 才算真被提起来


def sh(cmd, timeout=300):
    return subprocess.run(['bash', '-lc', SRC + cmd],
                          capture_output=True, text=True, timeout=timeout).stdout


def read_cube():
    """返回方块在世界系的 (x, y, z)，直接读 Gazebo 内部状态。"""
    out = sh('timeout 25 ign model -m target_cube -p', timeout=60)
    m = re.search(r'\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', out)
    if not m:
        print('!! 读不到方块位姿，输出如下：')
        print(out)
        sys.exit(1)
    return [float(m.group(i)) for i in (1, 2, 3)]


def send_arm(positions, dur, label):
    pts = ', '.join('%.6f' % v for v in positions)
    goal = ('{trajectory: {joint_names: [%s], points: [{positions: [%s], '
            'time_from_start: {sec: %d}}]}}' % (', '.join(JOINTS), pts, dur))
    out = sh('timeout %d ros2 action send_goal /arm_controller/follow_joint_trajectory '
             'control_msgs/action/FollowJointTrajectory "%s"'
             % (dur * 30 + 60, goal), timeout=dur * 30 + 90)
    ok = 'SUCCEEDED' in out
    err = next((l.strip() for l in out.splitlines() if 'error_string' in l), '')
    print(f'    {label:8s} -> {"SUCCEEDED" if ok else "ABORTED"}  {err}')
    return ok


def send_grip(pos, label):
    out = sh('timeout 200 ros2 action send_goal /gripper_controller/gripper_cmd '
             'control_msgs/action/GripperCommand '
             '"{command: {position: %.4f, max_effort: %.1f}}"' % (pos, GRIP_EFFORT),
             timeout=240)
    reached = 'reached_goal: true' in out
    stalled = 'stalled: true' in out
    print(f'    {label:8s} -> reached_goal={reached} stalled={stalled}')
    return reached


def main():
    print('=' * 64)
    print('夹具抓取闭环演示')
    print('=' * 64)

    p0 = read_cube()
    print(f'方块初始位姿: x={p0[0]:+.4f}  y={p0[1]:+.4f}  z={p0[2]:.4f}\n')

    print('① 张开夹爪')
    send_grip(GRIP_OPEN, '张开')

    print('② 回零位')
    send_arm(HOME, 5, '回零位')

    print('③ 移到预抓取位')
    send_arm(PREGRASP, 5, '预抓取')

    print('④ 下压到抓取位')
    send_arm(GRASP, 5, '下压')

    print('⑤ 闭合夹爪')
    send_grip(GRIP_CLOSE, '闭合')
    p_grip = read_cube()
    print(f'       闭合后方块 z = {p_grip[2]:.4f}（应仍在 {p0[2]:.4f} 附近）')

    print('⑥ 抬起')
    send_arm(LIFT, 6, '抬起')
    time.sleep(3)

    p1 = read_cube()
    dz = p1[2] - p0[2]
    print(f'\n抬起后方块位姿: x={p1[0]:+.4f}  y={p1[1]:+.4f}  z={p1[2]:.4f}')
    print(f'           相对初始抬升 Δz = {dz*1000:+.1f} mm')

    print('=' * 64)
    if dz > CUBE_LIFT_THRESHOLD:
        print(f'抓取成功：方块被提离地面 {dz*1000:.1f} mm')
    else:
        print(f'抓取失败：方块基本没动（Δz = {dz*1000:.1f} mm）')
        print('   可能原因：夹持力不足（滑脱）／方块被手指碰倒／位形没对准')
    print('=' * 64)

    print('\n⑦ 放回')
    send_arm(GRASP, 5, '回抓取位')
    send_grip(GRIP_OPEN, '张开')
    send_arm(HOME, 5, '回零位')

    p2 = read_cube()
    print(f'\n放回后方块位姿: x={p2[0]:+.4f}  y={p2[1]:+.4f}  z={p2[2]:.4f}')


if __name__ == '__main__':
    main()
