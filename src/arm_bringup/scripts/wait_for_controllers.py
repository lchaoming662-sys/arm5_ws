#!/usr/bin/env python3
"""wait_for_controllers.py — 「控制器就绪」门闩节点

## 它解决什么问题

想把 MoveIt（move_group + RViz）放在控制器**真正 active 之后**启动，
但仿真那一路的 spawner 是在 `arm_gazebo/gz_launch.py` 里、
用 `OnProcessExit` 一层层串起来的**被 include 的**动作——
外层 launch 没法用 `OnProcessExit(target_action=spawner)` 去挂它们
（`target_action` 只认本文件里定义的动作）。

不用固定延时（`TimerAction(period=30)`）是有理由的：本机 RTF 只有 0.39，
30 s 真实时间只等于约 12 s 仿真时间，机器一忙就不够；机器空一点又白等。
所以改成**问状态**：本节点反复调 `/controller_manager/list_controllers`，
直到点名的控制器全部 `state == 'active'` 就退出 0，由外层 launch 用
`OnProcessExit` 接住、再拉起 MoveIt。等多久取决于真的好了没有。

## 用法

    ros2 run arm_bringup wait_for_controllers.py \
        --controllers joint_state_broadcaster arm_controller gripper_controller \
        --reactivate gripper_controller \
        --timeout 180

退出码：0 = 全部 active（且重新申领已完成）；1 = 超时（会打印还差谁）；
        2 = 全部 active 了，但重新申领失败。
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import (ListControllers, SwitchController)

ACTIVE = 'active'


class ControllerWaiter(Node):

    def __init__(self, manager: str):
        super().__init__('wait_for_controllers')
        self._cli = self.create_client(
            ListControllers, f'{manager}/list_controllers')
        self._switch = self.create_client(
            SwitchController, f'{manager}/switch_controller')

    def service_ready(self, timeout: float) -> bool:
        return self._cli.wait_for_service(timeout_sec=timeout)

    def states(self, timeout: float):
        """调一次 list_controllers，返回 {控制器名: 状态}；失败返回 None。"""
        future = self._cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            return None
        return {c.name: c.state for c in future.result().controller}

    def _switch_state(self, activate, deactivate, timeout=10.0):
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = False
        future = self._switch.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            return False, '服务调用超时'
        return bool(future.result().ok), 'ok'

    def reactivate(self, names, period=1.0):
        """把指定控制器停掉再起来一次，让硬件重新申领它的命令接口。

        这一步是必需的，不是可选的（实测）：
        gz_ros2_control 只在【控制器申领接口】时才把关节的 joint_control_method
        置成 POSITION。仿真刚起那一轮，夹爪控制器的申领没有生效 —— 结果是
        right_claw_joint 完全不动：/joint_states 恒为 0，且无论用
        GripperActionController 还是 JointTrajectoryController 都一样。
        把控制器重新申领一次之后，关节立刻就听话了（实测闭合到 -0.200）。

        现象之所以难查，是因为控制器那一侧完全"正常"：goal 被接受、
        action 返回 SUCCEEDED，只有关节一动不动。
        """
        for name in names:
            ok, msg = self._switch_state(activate=[], deactivate=[name])
            if not ok:
                self.get_logger().error(f'停用 {name} 失败：{msg}')
                return False
            time.sleep(period)
            ok, msg = self._switch_state(activate=[name], deactivate=[])
            if not ok:
                self.get_logger().error(f'启用 {name} 失败：{msg}')
                return False
            self.get_logger().info(f'已让 {name} 重新申领一次命令接口')
            time.sleep(period)
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--controllers', nargs='+', required=True,
                        help='要等的控制器名（全部 active 才算就绪）')
    parser.add_argument('--manager', default='/controller_manager',
                        help='controller_manager 节点名')
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='总超时（秒）')
    parser.add_argument('--period', type=float, default=2.0,
                        help='轮询间隔（秒）')
    parser.add_argument(
        '--reactivate', nargs='*', default=[],
        help='全部就绪之后再把这些控制器停掉重起一次，让硬件重新申领命令接口。'
             '夹爪控制器必须做这一步，否则关节不动（详见本文件里的说明）')
    # launch 会往 argv 里塞 -r __node:=... -p xxx:=yyy 之类的 ROS 参数，
    # 用 parse_known_args 忽略掉，别让它们被当成错误参数。
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = ControllerWaiter(args.manager)
    deadline = time.monotonic() + args.timeout
    try:
        first = True
        while time.monotonic() < deadline:
            if node.service_ready(timeout=2.0):
                states = node.states(timeout=5.0)
                if states is not None:
                    missing = [c for c in args.controllers
                               if states.get(c) != ACTIVE]
                    if not missing:
                        node.get_logger().info(
                            '控制器已全部 active：' + '、'.join(args.controllers))
                        if args.reactivate:
                            if not node.reactivate(args.reactivate):
                                return 2
                        return 0
                    if first:
                        node.get_logger().info(
                            '等待控制器就绪（仿真刚起，这一步通常要 30~60 s）…')
                        first = False
                    detail = '、'.join(f'{c}={states.get(c)}' for c in missing)
                    node.get_logger().info(f'还差：{detail}')
            elif first:
                node.get_logger().info(
                    f'等待 {args.manager}/list_controllers 服务出现'
                    '（说明 gz_ros2_control 插件还没把 controller_manager 注册进来）…')
                first = False
            time.sleep(args.period)

        node.get_logger().error(
            f'超时 {args.timeout:.0f} s，控制器仍未全部 active：'
            + '、'.join(args.controllers)
            + '。先确认 Gazebo 是否真的起来了。')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
