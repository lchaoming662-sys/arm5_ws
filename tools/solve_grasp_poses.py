"""解出抓取序列的关节位形，并自检地面/基座碰撞。

用法：cd ~/arm5_ws/tools && python3 solve_grasp_poses.py
解出来的三个位形要手抄回 pick_demo.py 顶部的 HOME / PREGRASP / GRASP / LIFT。
"""
import math
from grasp_geometry import grasp_state, fk, dot, mv, add

UP = [0.0, 0.0, 1.0]
LIMIT = 1.5708
Q_GRASP = (-0.314, -0.244, -1.466)      # 扫描得到：指尖离地 4.6mm，已避开限位

S = grasp_state({'lower_arm_joint': Q_GRASP[0], 'upper_arm_joint': Q_GRASP[1], 'wrist_joint': Q_GRASP[2]})
TCP0 = S['tcp']
print(f"抓取位形 TCP = ({TCP0[0]:+.4f}, {TCP0[1]:+.4f}, {TCP0[2]:.4f})  指尖z={S['tip_z']:+.4f}  开口={S['gap']*1000:.2f}mm")
print(f"方块（25mm 放地面）中心 z = 0.0125，两指夹持范围 z ∈ [{S['tip_z']:.4f}, {S['tip_z']+0.037:.4f}]\n")


def err(q, target, w_o):
    s = grasp_state(q)
    e = sum((s['tcp'][i] - target[i]) ** 2 for i in range(3))
    if w_o:
        e += w_o * sum((s['plate_long'][i] - UP[i]) ** 2 for i in range(3))
    return e


def ik(target, q0, w_o=0.02, iters=6000):
    q = list(q0)
    step, cur = 0.10, err({'lower_arm_joint': q0[0], 'upper_arm_joint': q0[1], 'wrist_joint': q0[2]}, target, w_o)
    for _ in range(iters):
        moved = False
        for k in range(3):
            for d in (+1, -1):
                t = list(q)
                t[k] = max(-LIMIT, min(LIMIT, t[k] + d * step))
                e2 = err({'lower_arm_joint': t[0], 'upper_arm_joint': t[1], 'wrist_joint': t[2]}, target, w_o)
                if e2 < cur - 1e-12:
                    q, cur, moved = t, e2, True
        if not moved:
            step *= 0.5
            if step < 1e-7:
                break
    return q, cur


def report(q2, q3, q4, label):
    q = {'lower_arm_joint': q2, 'upper_arm_joint': q3, 'wrist_joint': q4}
    s = grasp_state(q)
    f = fk(q)
    tilt = math.degrees(math.acos(max(-1, min(1, dot(s['plate_long'], UP)))))
    print(f"[{label}]")
    print(f"  lower_arm_joint={q2:+.4f}  upper_arm_joint={q3:+.4f}  wrist_joint={q4:+.4f}   (top_plate_joint=0, claw_base_joint=0)")
    print(f"  TCP=({s['tcp'][0]:+.4f},{s['tcp'][1]:+.4f},{s['tcp'][2]:.4f})  指尖z={s['tip_z']:+.4f}  朝向偏差={tilt:.1f}°")
    warn = []
    for link in ['base_plate', 'top_plate', 'lower_arm', 'upper_arm', 'wrist', 'claw_base', 'right_claw']:
        z = f[link][1][2]
        if z < 0.005:
            warn.append(f"{link}.z={z:+.4f}")
    print(f"  地面/基座风险点: {'无' if not warn else warn}")
    return q, s


seq = {}
for label, dz, dy, w_o in [('抬起', +0.130, -0.010, 0.0),
                           ('预抓取', +0.045, -0.015, 0.02),
                           ('抓取', 0.0, 0.0, 0.02)]:
    tgt = [TCP0[0], TCP0[1] + dy, TCP0[2] + dz]
    q, e = ik(tgt, Q_GRASP, w_o)
    _, s = report(q[0], q[1], q[2], f"{label}  目标z={tgt[2]:.4f}  残差={e:.2e}")
    seq[label] = q
    print()

print("=== 汇总（写进 pick 脚本用）===")
for k, v in seq.items():
    print(f"  {k:6s}: [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")
print(f"  home  : [0.0, 0.0, 0.0]")
