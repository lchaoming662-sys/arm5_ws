#!/usr/bin/env python3
"""ik_probe.py — 对指定目标做 /compute_ik 单点探针（对比测试用）"""
import json
import sys
import time

import rclpy
from rclpy.duration import Duration
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState as RobotStateMsg, PositionIKRequest
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

HARD_LIMIT = 1.5708
ARM_JOINTS = ['top_plate_joint', 'lower_arm_joint', 'upper_arm_joint',
              'wrist_joint', 'claw_base_joint']
KEY = ['wrist_joint', 'claw_base_joint']

TARGETS = [
    {'x': 0.0267, 'y': 0.12,  'z': 0.20},   # 踩坑 17 条验证过的可达目标
    {'x': 0.0267, 'y': 0.12,  'z': 0.25},
    {'x': 0.0267, 'y': 0.05,  'z': 0.20},
    {'x': 0.0267, 'y': -0.05, 'z': 0.20},
]


def main():
    rclpy.init()
    node = rclpy.create_node('ik_probe')
    js_pub = node.create_publisher(JointState, '/joint_states', 10)

    def pub_js():
        m = JointState()
        m.header.stamp = node.get_clock().now().to_msg()
        m.name = ARM_JOINTS + ['right_claw_joint']
        m.position = [0.0] * len(m.name)
        js_pub.publish(m)

    timer = node.create_timer(0.1, pub_js)
    cli = node.create_client(GetPositionIK, '/compute_ik')
    cli.wait_for_service(timeout_sec=10.0)
    time.sleep(1.0)   # 等 CurrentStateMonitor 吃到状态

    for t in TARGETS:
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = 'arm'
        req.ik_request.ik_link_name = 'tcp'
        req.ik_request.timeout = Duration(seconds=0.5).to_msg()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped.header.frame_id = 'world'
        req.ik_request.pose_stamped.pose = Pose(
            position=Point(x=t['x'], y=t['y'], z=t['z']),
            orientation=Quaternion(w=1.0))
        seed = JointState()
        seed.name = ARM_JOINTS + ['right_claw_joint']
        seed.position = [0.0] * len(seed.name)
        req.ik_request.robot_state = RobotStateMsg(joint_state=seed)

        t0 = time.perf_counter()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        dt = (time.perf_counter() - t0) * 1000
        r = fut.result()
        if r is None:
            print(f"y={t['y']:+.2f} z={t['z']:.2f}  调用超时")
            continue
        if r.error_code.val == 1:
            sol = dict(zip(r.solution.joint_state.name,
                           r.solution.joint_state.position))
            ms = ', '.join(f"{j.replace('_joint','')}: value={sol[j]:+.4f} "
                           f"margin={HARD_LIMIT - abs(sol[j]):.4f}" for j in KEY)
            print(f"y={t['y']:+.2f} z={t['z']:.2f}  err=+1  t={dt:6.1f}ms  {ms}")
        else:
            print(f"y={t['y']:+.2f} z={t['z']:.2f}  err={r.error_code.val:+d}  t={dt:6.1f}ms")

    timer.cancel()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
