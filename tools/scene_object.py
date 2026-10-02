#!/usr/bin/env python3
"""scene_object.py — 往 move_group 的规划场景里加 / 删目标方块

为什么必须单独有这一步：
  仿真里的 target_cube 只活在 Gazebo 的物理世界里，MoveIt 的 planning scene
  对它一无所知。而 MTC 的 GenerateGraspPose 是从 planning scene 里查物体来
  生成候选的 —— 场景里没有它，就直接报
      "object 'target_cube' not in scene"
  而且必须在**规划开始之前**加好：MTC 的 scene() 给的是初始场景，任务内部的
  ModifyPlanningScene 只修改向下传播的状态、改不到它。所以「在 task 里加物体
  再生成候选」这个写法是不成立的。

  这也是 MTC 官方 demo 用 moveit_commander 的 PlanningSceneInterface 在外面
  加物体的原因。本机没装 moveit_commander，这里直接用 /apply_planning_scene
  服务，效果等价。

用法：
    /usr/bin/python3 ~/arm5_ws/tools/scene_object.py --add
    /usr/bin/python3 ~/arm5_ws/tools/scene_object.py --remove
"""
import argparse
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

OBJECT_ID = 'target_cube'
DEFAULT_XYZ = [0.0267, 0.0245, 0.0125]   # 世界文件里的真值，贴地 25 mm 方块
DEFAULT_SIZE = [0.025, 0.025, 0.025]


def main():
    ap = argparse.ArgumentParser(description='把目标方块加进 move_group 的规划场景')
    ap.add_argument('--add', action='store_true', help='加入场景')
    ap.add_argument('--remove', action='store_true', help='从场景移除')
    ap.add_argument('--xyz', nargs=3, type=float, default=DEFAULT_XYZ,
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--size', nargs=3, type=float, default=DEFAULT_SIZE,
                    metavar=('DX', 'DY', 'DZ'))
    args = ap.parse_args()

    if not args.add and not args.remove:
        print('要指定 --add 或 --remove', file=sys.stderr)
        return 2

    rclpy.init()
    node = Node('scene_object')

    co = CollisionObject()
    co.header.frame_id = 'base_link'
    co.id = OBJECT_ID

    if args.add:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in args.size]
        co.primitives.append(box)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in args.xyz]
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
    else:
        co.operation = CollisionObject.REMOVE

    scene = PlanningScene()
    scene.world.collision_objects.append(co)
    scene.is_diff = True   # 差分更新：只动这一个物体，不覆盖场景其余部分

    cli = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    if not cli.wait_for_service(timeout_sec=10.0):
        print('等不到 /apply_planning_scene —— move_group 没在跑？', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 3

    req = ApplyPlanningScene.Request()
    req.scene = scene
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    if future.result() is None:
        print('服务调用失败', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 4
    if not future.result().success:
        print('move_group 拒绝了这次场景更新', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 5

    action = '加入' if args.add else '移除'
    print(f'已{action}场景物体 {OBJECT_ID}'
          + (f'  中心 {args.xyz}  尺寸 {args.size}' if args.add else ''))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
