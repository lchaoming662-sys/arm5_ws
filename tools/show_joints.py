"""打印一次 /joint_states 的关节名与位置（顺序对齐，避免读错）。

用 ros2 topic echo 的文本输出很容易把关节对错位（本工程的 /joint_states 顺序是
[top_plate, upper_arm, lower_arm, wrist, claw_base, right_claw, left_claw_mimic]
—— 注意 upper_arm 在 lower_arm 前面），所以这里直接按 name 对齐打印。
"""
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState


def main():
    rclpy.init()
    node = Node('show_joints',
                parameter_overrides=[Parameter('use_sim_time',
                                               Parameter.Type.BOOL, True)])
    got = {}
    node.create_subscription(JointState, '/joint_states',
                             lambda m: got.setdefault('m', m), 5)
    import time
    t0 = time.time()
    while 'm' not in got and time.time() - t0 < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'm' not in got:
        print('没收到 /joint_states')
        return 2
    m = got['m']
    for n, p in zip(m.name, m.position):
        print(f'  {n:26s} {p:+.6f}')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
