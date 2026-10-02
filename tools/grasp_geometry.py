"""arm5 抓取几何工具：FK / 夹爪位姿 / 简易 IK

展开后的 URDF 需要一份。优先用工作区里的，找不到就现场从 xacro 展开一次。
（改过 arm_core.xacro 后请重跑本模块，避免用旧模型算出来的位形。）
"""
import math
import os
import subprocess
import xml.etree.ElementTree as ET

_WS = os.path.expanduser('~/arm5_ws')
_XACRO = os.path.join(_WS, 'install/arm_description/share/arm_description/urdf/arm_gz.urdf.xacro')


def _find_urdf():
    for p in ('/tmp/arm_gz.urdf',
              os.path.join(_WS, 'install/arm_description/share/arm_description/urdf/arm_gz.urdf.urdf')):
        if os.path.isfile(p):
            return p
    subprocess.run(['bash', '-lc',
                    'source /opt/ros/humble/setup.bash && '
                    f'source {_WS}/install/setup.bash && '
                    f'xacro {_XACRO} > /tmp/arm_gz.urdf'],
                   capture_output=True, text=True)
    return '/tmp/arm_gz.urdf'


URDF = _find_urdf()

BOX = {  # 夹爪碰撞盒：相对各自 jaw link 的 origin 与尺寸
    'right_claw': ([-0.015500, -0.004500, -0.013500], [0.012, 0.012, 0.037]),
    'left_claw': ([0.015700, -0.004500, -0.013500], [0.012, 0.012, 0.037]),
}
JAW_HALF_THICK = 0.006      # 盒在 sep 方向的半厚 (0.012/2)


def R_rpy(rpy):
    x, y, z = rpy
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return [[cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
            [cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
            [-sy, cy*sx, cx*cy]]


def R_axis(a, th):
    x, y, z = a
    n = math.sqrt(x*x + y*y + z*z)
    x, y, z = x/n, y/n, z/n
    c, s, C = math.cos(th), math.sin(th), 1 - math.cos(th)
    return [[x*x*C + c, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]]


def mul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv(R, v):
    return [sum(R[i][k]*v[k] for k in range(3)) for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 1e-12 else [0.0, 0.0, 0.0]


def dot(a, b):
    return sum(a[i]*b[i] for i in range(3))


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


_root = ET.parse(URDF).getroot()
_kids = {}
for _j in _root.findall('joint'):
    _kids.setdefault(_j.find('parent').get('link'), []).append(_j)

REVOLUTE = ['top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 'wrist_joint', 'claw_base_joint']


def fk(q):
    out = {}

    def walk(link, R, t):
        out[link] = (R, t)
        for j in _kids.get(link, []):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()]
            Rn, tn = mul(R, R_rpy(rpy)), add(t, mv(R, xyz))
            if j.get('type') in ('revolute', 'continuous'):
                ax = [float(v) for v in j.find('axis').get('xyz').split()]
                Rn = mul(Rn, R_axis(ax, q.get(j.get('name'), 0.0)))
            walk(j.find('child').get('link'), Rn, tn)

    walk('world', [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])
    return out


def grasp_state(q):
    """返回 tcp / 指长轴 / 开合轴 / 两指接触面中心 / 面间距 / 手指最低点"""
    f = fk(q)
    ja, th_a = f['right_claw']
    jb, th_b = f['left_claw']
    ca = add(th_a, mv(ja, BOX['right_claw'][0]))
    cb = add(th_b, mv(jb, BOX['left_claw'][0]))
    plate_long = norm(mv(ja, [0, 0, 1]))          # jaw 局部 +Z = 手指长轴
    sep = norm(sub(cb, ca))
    tcp = [(ca[i] + cb[i]) / 2.0 for i in range(3)]
    face_a = add(ca, [JAW_HALF_THICK * sep[i] for i in range(3)])
    face_b = sub(cb, [JAW_HALF_THICK * sep[i] for i in range(3)])
    gap = math.sqrt(sum(x*x for x in sub(face_b, face_a)))
    # 手指沿长轴的半长 = 0.037/2
    tips = min(add(ca, [i * (0.037 / 2) for i in plate_long])[2],
               sub(ca, [i * (0.037 / 2) for i in plate_long])[2],
               add(cb, [i * (0.037 / 2) for i in plate_long])[2],
               sub(cb, [i * (0.037 / 2) for i in plate_long])[2])
    return {'tcp': tcp, 'plate_long': plate_long, 'sep': sep,
            'face_a': face_a, 'face_b': face_b, 'gap': gap,
            'tip_z': tips, 'jaw_a_c': ca, 'jaw_b_c': cb}


if __name__ == '__main__':
    print("=== 零位 ===")
    s = grasp_state({})
    print(" TCP        ", [round(v, 4) for v in s['tcp']])
    print(" 指长轴     ", [round(v, 3) for v in s['plate_long']])
    print(" 开合轴     ", [round(v, 3) for v in s['sep']])
    print(" 面间距 mm  ", round(s['gap'] * 1000, 2))

    print("\n=== 扫描：找出「两指竖直朝下」的俯仰配置 ===")
    DOWN = [0.0, 0.0, -1.0]
    found = []
    N = 60
    for i2 in range(N + 1):
        q2 = -math.pi / 2 + math.pi * i2 / N
        for i3 in range(N + 1):
            q3 = -math.pi / 2 + math.pi * i3 / N
            for i4 in range(N + 1):
                q4 = -math.pi / 2 + math.pi * i4 / N
                q = {'lower_arm_joint': q2, 'upper_arm_joint': q3, 'wrist_joint': q4}
                st = grasp_state(q)
                if dot(st['plate_long'], DOWN) > 0.995:
                    found.append((q2, q3, q4, st['tcp'][2], st['tip_z'], st['tcp'][1]))
    print(f" 命中 {len(found)} 个配置")
    if found:
        zs = [f[3] for f in found]
        print(f" TCP z 范围: [{min(zs):.4f}, {max(zs):.4f}]")
        lo = sorted(found, key=lambda f: f[3])[:3]
        hi = sorted(found, key=lambda f: -f[3])[:3]
        print(" 最低的 3 个:")
        for q2, q3, q4, z, tz, y in lo:
            print(f"   q2={q2:+.3f} q3={q3:+.3f} q4={q4:+.3f}  TCP_z={z:.4f} 指最低点={tz:+.4f} TCP_y={y:+.4f}")
        print(" 最高的 3 个:")
        for q2, q3, q4, z, tz, y in hi:
            print(f"   q2={q2:+.3f} q3={q3:+.3f} q4={q4:+.3f}  TCP_z={z:.4f} 指最低点={tz:+.4f} TCP_y={y:+.4f}")
