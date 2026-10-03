#!/usr/bin/env python3
"""estimate_object_pose.py — 从腕部 RGBD 点云估计目标物体位姿

这是抓取流水线的感知入口：回答「相机看到的东西在哪、多大」。
输出（中心 + 尺寸）直接喂给抓取候选生成。

前提：
  · 仿真正在跑（`ros2 launch arm_gazebo gz_launch.py`）
  · 机器人已经摆到看得见目标的位形。注意零位时光轴是水平朝前的
    （实测与竖直夹角 85.2 度），地面上的东西根本不在视野里；
    用 SRDF 里的 grasp_ready 预设即可 —— 实测偏轴 12.3 度、距离 106 mm。

用法：
    /usr/bin/python3 ~/arm5_ws/tools/estimate_object_pose.py
    加 --model target_cube 会顺便读 Gazebo 内部真值做对照。
    加 --json 则只输出一行机器可读结果（供抓取流水线消费），人类可读
    的过程输出全部丢弃。

分割思路：
  视野里绝大多数点是地面，所以先取 z 的直方图峰值当地面高度，把地面附近
  的点剔掉，剩下的做体素连通域聚类，取最大的一簇当目标。

有效性判据（本轮改动，见 tools/geometry_validity.py）：
  聚类之后**不再**用「跨度」判有效性，而是做前置拒绝的多重几何验证：
    ① 面残差（扣除观察倾斜后）< 4 mm  —— 它是一块**面**，不是一堆乱点
    ② 主平面上凸包面积比∈ [0.6, 2.0] —— 它**只占**这么大地方
    ③ 法向一致性 E[1-|cosθ|] < 0.25   —— 表面朝向一致
  三项全过才输出位姿，任一不过直接退出（退出码 8）。
  中心估计改为MCD（最小协方差行列式）剔离群 + p1p99 分位。

  为什么换掉跨度判据：跨度只度量「看到了多宽」，对「这是不是一块
  **完整**的物体」没有任何判断力。实测（docs/踩坑记录.md 第 42 条）
  目标被手指挡掉一半时，碎片在 x 方向仍有 25 mm 宽，跨度放行而中心
  偏了 3~4 mm；更糟的��况下偏 10 mm，流水线照用。

  阈值的标定过程见 tools/calib/ 下两个脚本。**改阈值前必须重跑它们** ——
  判据数值来自实测，不是拍的。

注意（两条既有的坑，都保留）：
  注意点云是在相机系里的，单位是米，但相机装在腕上会随臂运动，
  所以先经 TF 转到 base_link 再做几何判断。真值对照也在base_link 下做，
  两边参考系必须一致，否则误差里混进的是坐标系差异而不是算法误差。
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
import tf2_ros

# 同目录的几何工具。导入时它会展开一次 xacro（约 1 秒），换来的是两指碰撞盒
# 的精确位置 —— 自身遮挡剔除要用，那是唯一能把手指和贴地目标分开的依据：
# 俯抓姿态下指尖离方块顶面只有 1.8 mm，离镜头的距离也几乎一样。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grasp_geometry import fk as arm_fk, BOX as CLAW_BOX  # noqa: E402
from geometry_validity import validate_and_estimate  # noqa: E402

POINTS_TOPIC = '/wrist_cam/points'
JOINTS_TOPIC = '/joint_states'

# 点云数据实际所在的参考系 —— 注意它不是消息 header 里写的那个。
#
# Gazebo Fortress 的 rgbd_camera 把点云表达在传感器 link 自身的 body 系
# （+X 前 / +Y 左 / +Z 上），而 <gz_frame_id> 只是把消息的 frame_id 盖成
# optical 系（+Z 前 / +X 右 / +Y 下），两套轴向差着一个固定旋转。
# 照 header.frame_id 查 TF，整片点云会被转错 90 度：实测地面会从 z = 0
# 跑到 z = 0.13，而且不报任何错 —— 又一处「安静地给错答案」。
#
# 图像不受影响（二维图拿 optical 系当 frame_id 才是对的），所以也不能靠改
# <gz_frame_id> 一劳永逸：那个参数是传感器级、四个输出共用的。
# 这个不一致没法在模型里消除，只能在这里留下唯一的处理点。
POINTS_DATA_FRAME = 'wrist_cam_link'


def quat_to_matrix(x, y, z, w):
    """四元数转旋转矩阵（行主序，作用方式 v' = R @ v）。"""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


def cloud_to_xyz(msg):
    """把 PointCloud2 解成 (N,3) float32。字段偏移从消息里读，不写死。"""
    off = {f.name: f.offset for f in msg.fields}
    for need in ('x', 'y', 'z'):
        if need not in off:
            raise RuntimeError(f'点云里没有 {need} 字段，实际有 {sorted(off)}')
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    def col(name):
        o = off[name]
        return buf[:, o:o + 4].copy().view(np.float32).ravel()

    return np.stack([col('x'), col('y'), col('z')], axis=1)


def segment(xyz, ground_tol, voxel, min_pts, verbose=True):
    """剔地面 → 体素连通域聚类，返回（簇列表，地面高度，有效点数）。"""
    pts = xyz[np.isfinite(xyz).all(axis=1)]
    n_valid = len(pts)
    if n_valid == 0:
        return [], None, 0

    # 地面高度 = z 直方图峰值。视野里地面占绝对多数，峰值就是它。
    zs = pts[:, 2]
    hist, edges = np.histogram(zs, bins=200)
    peak = int(hist.argmax())
    z_ground = 0.5 * (edges[peak] + edges[peak + 1])

    above = pts[np.abs(pts[:, 2] - z_ground) > ground_tol]
    if verbose:
        print(f'  有效点 {n_valid}，地面高度 z = {z_ground:+.4f}，'
              f'剔地面后剩 {len(above)} 点')
    if len(above) == 0:
        return [], z_ground, n_valid

    # 体素化后用 26 邻域连通域。先生成紧凑占据网格，再 label，
    # 最后把每个点映回它所在体素的标签。
    keys = np.floor(above / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    shape = keys.max(axis=0) + 1
    occ = np.zeros(shape, dtype=bool)
    occ[keys[:, 0], keys[:, 1], keys[:, 2]] = True
    lab, n_lab = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=bool))
    labels = lab[keys[:, 0], keys[:, 1], keys[:, 2]]

    clusters = []
    for i in range(1, n_lab + 1):
        m = labels == i
        if int(m.sum()) >= min_pts:
            clusters.append(above[m])
    clusters.sort(key=len, reverse=True)
    return clusters, z_ground, n_valid


def self_filter(points_base, fk_result, margin):
    """剔除落在两指碰撞盒内的点，返回保留掩码。

    腕部相机必然拍得到自己的两指，而俯抓姿态下指尖和贴地目标在几何上几乎
    贴在一起：实测指尖 z = 0.0268、方块顶面 z = 0.025，离镜头都是 90 mm 上下。
    靠高度阈值或距离阈值都分不开。但手指在哪，我们是精确知道的（URDF + FK），
    所以直接按几何剔除 —— 这是自遮挡处理最干净的做法，也不依赖调参。

    margin 是盒的膨胀量，用来吃掉深度噪声（噪声实测 std 约 1.3 mm）。
    """
    keep = np.ones(len(points_base), dtype=bool)
    for link, (org, size) in CLAW_BOX.items():
        if link not in fk_result:
            continue
        R, t = fk_result[link]
        Rm = np.asarray(R, dtype=float)
        tm = np.asarray(t, dtype=float)
        # fk 约定：v_world = R @ v_link + t，所以 v_link = R^T @ (v_world - t)
        local = (points_base - tm) @ Rm
        half = np.asarray(size, dtype=float) / 2.0 + margin
        inside = np.all(np.abs(local - np.asarray(org, dtype=float)) <= half, axis=1)
        keep &= ~inside
    return keep


def gazebo_truth(model):
    """读 Gazebo 内部位姿。返回 (x,y,z) 或 None。

    ign 的输出是多行的，开头还有一行 `Model: [8]` —— 那个 8 是模型 ID，
    不是坐标。只认「整行是一个方括号、里面恰好三个数」的那种行，
    否则会把模型 ID 当成 x 读进来。
    """
    try:
        r = subprocess.run(['ign', 'model', '-m', model, '-p'],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in r.stdout.splitlines():
        s = line.strip()
        if not (s.startswith('[') and s.endswith(']')):
            continue
        try:
            vals = [float(v) for v in s[1:-1].split()]
        except ValueError:
            continue
        if len(vals) == 3:
            return vals
    return None


def main():
    ap = argparse.ArgumentParser(description='从腕部 RGBD 点云估计目标物体位姿')
    ap.add_argument('--frame', default='base_link',
                    help='输出参考系（默认 base_link）')
    ap.add_argument('--ground-tol', type=float, default=0.004,
                    help='地面带半宽，单位米（默认 4 mm）')
    ap.add_argument('--voxel', type=float, default=0.002,
                    help='聚类体素边长，单位米（默认 2 mm）')
    ap.add_argument('--min-pts', type=int, default=30,
                    help='成簇最少点数（默认 30）')
    ap.add_argument('--self-margin', type=float, default=0.004,
                    help='两指碰撞盒的膨胀量，用于自身遮挡剔除（默认 4 mm）')
    ap.add_argument('--z-max', type=float, default=0.035,
                    help='只保留低于该高度的点（base_link 系，默认 0.035 m）。'
                         '依据是这条臂只能抓地面上的物体；设 -1 关闭')
    ap.add_argument('--model', default=None,
                    help='顺便读该 Gazebo 模型的真值做对照，如 target_cube')
    ap.add_argument('--expect-size', type=float, default=0.025,
                    help='预期工件宽度（米），用于有效性检查。设 0 关闭检查。'
                         '默认 25 mm —— 本工程的目标是 25 mm 立方体')
    ap.add_argument('--min-span-ratio', type=float, default=0.80,
                    help='实测跨度不得小于 工件尺寸×该比例（默认 0.80）')
    ap.add_argument('--max-span-ratio', type=float, default=1.80,
                    help='实测跨度不得大于 工件尺寸×该比例（默认 1.80）')
    ap.add_argument('--min-points-total', type=int, default=400,
                    help='点簇点数下限，低于此值判为没看全（默认 400）')
    ap.add_argument('--json', action='store_true',
                    help='只输出一行 JSON（给抓取流水线消费），'
                         '人类可读的过程输出全部丢弃')
    args = ap.parse_args()

    # --json 模式下把过程 print 全部导进黑洞，只留最后那一行结果。
    # 用 stdout 重定向而不是逐个改 print，是为了不动那 20 多行诊断输出 ——
    # 那些内容在调试时很有用，不该为了机器可读而牺牲掉。
    real_stdout = sys.stdout
    if args.json:
        sys.stdout = io.StringIO()

    def finish(code, payload=None):
        """收尾：恢复 stdout，并按需吐出那一行 JSON。"""
        sys.stdout = real_stdout
        if payload is not None:
            print(json.dumps(payload, ensure_ascii=False))
        return code

    rclpy.init()
    node = Node('estimate_object_pose',
                parameter_overrides=[Parameter('use_sim_time', value=True)])
    cloud = {}
    joints = {}
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    sub = node.create_subscription(
        PointCloud2, POINTS_TOPIC, lambda m: cloud.__setitem__('msg', m), qos)
    sub_j = node.create_subscription(
        JointState, JOINTS_TOPIC, lambda m: joints.__setitem__('msg', m), qos)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    print(f'等待 {POINTS_TOPIC} 与 {JOINTS_TOPIC} ...')
    t0 = time.time()
    while ('msg' not in cloud or 'msg' not in joints) and time.time() - t0 < 20.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'msg' not in cloud:
        print('超时：没收到点云。检查仿真在跑、相机没被 camera:=false 关掉。',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(2)
    if 'msg' not in joints:
        print('超时：没收到 /joint_states，无法按 URDF 剔除自身遮挡。',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(2)
    sub.destroy()
    sub_j.destroy()

    msg = cloud['msg']
    xyz = cloud_to_xyz(msg)
    print(f'收到点云：{msg.width}x{msg.height} = {len(xyz)} 点')
    print(f'  消息自称 frame_id = {msg.header.frame_id}，'
          f'但数据实际在 {POINTS_DATA_FRAME}（见脚本内说明）')

    # 转参考系。两个坑：
    #  ① 源 frame 用 POINTS_DATA_FRAME 而不是 msg.header.frame_id —— 后者撒谎，
    #     理由见文件上方关于 POINTS_DATA_FRAME 的说明。
    #  ② 用「最新可用」的变换，而不是点云自己的时间戳：两条发布链各走各的，
    #     消息戳常比 TF 最新时间新十几毫秒，按消息戳查会直接报
    #     "extrapolation into the future"。相机刚性装在腕上、又在静止位形下
    #     观测，用最新变换不引入额外误差。
    try:
        tr = tf_buffer.lookup_transform(args.frame, POINTS_DATA_FRAME,
                                        Time(),
                                        timeout=Duration(seconds=3.0))
    except Exception as exc:  # noqa: BLE001 —— 查不到 TF 的原因很多，统一报出来
        print(f'查不到 {POINTS_DATA_FRAME} -> {args.frame} 的 TF：{exc}',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(3)
    t = tr.transform.translation
    q = tr.transform.rotation
    R = quat_to_matrix(q.x, q.y, q.z, q.w)
    xyz = xyz @ R.T + np.array([t.x, t.y, t.z])

    # 自身遮挡剔除：按 URDF 算出的两指碰撞盒，剔掉落在盒内的点。
    # fk 输出的坐标就是 base_link 系（URDF 里 world->base_link 是零位固定关节），
    # 所以只在参考系取 base_link 时成立。
    if args.frame == 'base_link':
        jmsg = joints['msg']
        fk_res = arm_fk(dict(zip(jmsg.name, jmsg.position)))
        keep = self_filter(xyz, fk_res, args.self_margin)
        print('\n=== 自身遮挡剔除 ===')
        print(f'  落在两指碰撞盒内（膨胀 {args.self_margin*1000:.0f} mm）的点 '
              f'{int((~keep).sum())} 个，已剔除')
        xyz = xyz[keep]

        # 再按高度兜一刀。两指能按 URDF 精确剔除，但腕、夹爪底座和手臂都是
        # 复杂 mesh，没法同样处理；而这条臂的物理约束是「只能抓地面上的东西」
        # （见 docs/使用说明书.md 第七节），所以高于目标的一切都算干扰。
        if args.z_max > 0:
            high = xyz[:, 2] > args.z_max
            if high.any():
                print(f'  高于 z = {args.z_max:.3f} m 的点 {int(high.sum())} 个'
                      f'（自己的腕与臂），已剔除')
            xyz = xyz[~high]
    else:
        print(f'\n注意：参考系取的是 {args.frame} 而不是 base_link，'
              '跳过自身遮挡剔除（fk 只在 base_link 系下可用）')

    print(f'\n=== 分割（{args.frame} 系）===')
    clusters, z_ground, n_valid = segment(xyz, args.ground_tol,
                                          args.voxel, args.min_pts)
    if not clusters:
        print('没找到候选物体。可能目标不在视野、或点数太少。', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(4)
    print(f'  聚成 {len(clusters)} 簇，取最大的一簇')

    obj = clusters[0]

    # ==================================================================
    # 前置拒绝的多重几何验证（tools/geometry_validity.py）
    # ==================================================================
    # 下面这一段取代了原先的「跨度判据」（min-span-ratio / max-span-ratio）。
    #
    # 为什么换掉跨度判据：
    #   跨度只度量「看到了多宽」，它对两件事一无所知 ——
    #     ① 这块点云是不是一个**完整**的物体。目标被手指挡掉一半，
    #        剩下的碎片在 x 方向仍可能有 25 mm 宽，跨度放行，
    #        而中心已经偏了 3~4 mm（实测，见 docs/踩坑记录.md 第 42 条）。
    #     ② 这块点云是不是**一个**物体。粘连时跨度暴涨，跨度判据
    #        用「过大」那一侧挡，但那是唯一一次侥幸：粘连发生在
    #        厚度方向时跨度依然完美。
    #   三重验证（平面性 / 凸包面积 / 法向一致性）各自堵一个洞，
    #   且互相不相关。中心估计也从 bbox 中心换成 MCD + p1p99。
    #
    # 阈值与失效边界都标定在 tools/calib/ 里，改任何一个之前
    # 请重跑那些脚本 —— 判据的数值来自实测，不是拍脑袋。
    print(f'\n=== 几何验证（{args.frame} 系）===')
    try:
        vres = validate_and_estimate(obj, args.expect_size, frame=args.frame)
    except Exception as exc:                      # noqa: BLE001
        # 验证模块自己抛错 = 几何不合格，是**正常的业务结果**
        # （目标没看全，换个位置再来），不是程序故障。
        print(f'  ✗ {exc}', file=sys.stderr)
        print('    解决办法：把目标移到视野中心附近，或调整扫描位形。'
              '不要在结果可疑时继续用。', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(8)

    print(f'  {vres.summary()}')
    for name, (ok, val, thr) in vres.checks.items():
        # val 可能是 float（面残差）也可能 int（样本数），统一转 str 再对齐，
        # 否则 int 撞上 '>18s' 直接 ValueError（全链路回归 2026-10-03 实测）
        print(f'    {"✓" if ok else "✗"} {name:20s} 实测 {str(val):>18}'
              f'   要求 {thr}')
    if not vres.ok:
        print(f'\n  ✗ 几何验证未通过 —— 拒绝输出位姿。', file=sys.stderr)
        print(f'    原因：{vres.reason}', file=sys.stderr)
        print('    这正是第 42 条那个「静默给出偏 10 mm 位姿」的场景：'
              '宁可任务失败，也不能让机械臂扑空后报「执行完成」。',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return finish(8)

    # 通过验证后才取中心。**顺序不能颠倒** —— 先算中心再验证的话，
    # 一个「其实是碎片」的簇也会算出一个精确的数值进入下游，
    # 看起来比直接失败更可信。
    center = vres.center[:2]
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    size = np.array([np.percentile(obj[:, 0], 99.0) - np.percentile(obj[:, 0], 1.0),
                     np.percentile(obj[:, 1], 99.0) - np.percentile(obj[:, 1], 1.0),
                     hi[2] - lo[2]])

    print(f'\n=== 目标位姿（{args.frame} 系）===')
    print(f'  点数    {len(obj)}')
    print(f'  中心 xy x {center[0]:+.4f}   y {center[1]:+.4f}'
          f'   （MCD 内点 + p1p99 分位，非包围盒中心）')
    print(f'  观测范围 dx {size[0]:.4f}  dy {size[1]:.4f}  dz {size[2]:.4f}')
    print(f'  z 区间   [{lo[2]:+.4f}, {hi[2]:+.4f}]')
    print(f'  包围盒   x [{lo[0]:+.4f}, {hi[0]:+.4f}]  y [{lo[1]:+.4f}, {hi[1]:+.4f}]'
          f'   （对比用）')

    # ---------------------------------------------------------------------
    # 有效性检查已上移到几何验证那一步（validate_and_estimate）。
    #
    # 原先这里是「跨度 ∈ 工件尺寸 × [0.8, 1.8]」，现已删除。删掉的理由
    # 写在上面几何验证处的注释里：一句话 —— 跨度对「这是不是一块完整
    # 的物体」没有任何判断力，而那恰恰是第 42 条踩的坑。
    #
    # 保留 --min-span-ratio / --max-span-ratio 两个参数只为**兼容旧命令行**，
    # 它们不再影响判定。删掉参数会让任何还在传它的脚本直接报错 ——
    # 那比「传了但没生效」更安全，但会让标定脚本莫名其妙崩掉。
    # 等确认没有脚本再传它们之后删。
    # ---------------------------------------------------------------------
    if args.min_span_ratio != 0.80 or args.max_span_ratio != 1.80:
        print('注意：--min-span-ratio / --max-span-ratio 已不再参与判定'
              '（跨度判据已被多重几何验证取代），这两个值被忽略。',
              file=sys.stderr)

    # 相机几乎垂直向下（实测光轴与竖直只差 8.5 度），25 mm 高的侧面在图像里
    # 只摊开 25*sin(12.3) ≈ 5 mm —— 于是点云实际只覆盖了目标顶面，中心高度
    # 根本测不到。只能靠「物体贴地」这个前提反推：平顶物体贴地放置时，
    # 顶面离地高度就等于它的高度。这条臂本来也够不着台面上的东西
    # （见 docs/使用说明书.md 第七节），所以这个前提在本工程里始终成立。
    center_z = None
    if args.frame == 'base_link' and z_ground is not None:
        # 顶面高度怎么取，决定了中心 z 的精度，这里有讲究。
        #
        # 直接取 max 会**系统性偏高**：相机光轴偏竖直 12.3 度（不是正下方），
        # 于是同一个水平顶面在点云里是斜的 —— 远边的 z 比近边高。取 max 等于
        # 专门去用「远边那条棱」，而那条棱到相机的距离最长、深度噪声最大。
        # 实测 25 mm 的方块，max 给出离地 27.7 mm，偏高 2.7 mm。
        #
        # 这 2.7 mm 会一路传下去：中心 z 偏高 → 抓取点相对物体压低 → 手指
        # 多压 2.7 mm。而 docs/踩坑记录.md 第 27 条记着实测多压 1.7 mm 就能
        # 把方块挤走 11 mm，所以这不是小数点后面的事。
        #
        # 改用 90 分位：它落在顶面那片区域里，又避开了最外圈那条棱和长尾噪声。
        # 为什么不是更低：分位取得太低会切进顶面内部，反而低估。
        # 之所以敢用分位数，是因为这一簇是**平面分割 + 连通域**的结果，
        # 主体就是顶面本身，不是侧面或杂散点。
        z_top_max = float(hi[2])
        z_top_p90 = float(np.percentile(obj[:, 2], 90.0))
        z_top = z_top_p90
        center_z = (z_ground + z_top) / 2
        print(f'  顶面高度 max {z_top_max:+.4f} / 90分位 {z_top_p90:+.4f}'
              f'   离地高度 {(z_top - z_ground)*1000:.1f} mm（取 90 分位）')
        print(f'  贴地反推的中心 z = {center_z:+.4f}'
              '（相机几乎垂直向下，测不到中心高度）')

    truth = None
    if args.model:
        # 只调一次：ign 是子进程调用，第二次可能因为仿真瞬时忙而返回 None，
        # 于是 payload 里会出现 truth 有值、gazebo_truth 是 null 的怪组合。
        truth = gazebo_truth(args.model)
        print(f'\n=== 与 Gazebo 真值对照（{args.model}）===')
        if truth is None:
            print('  读不到真值（ign 命令不可用或模型名不对）')
        else:
            print(f'  真值中心  x {truth[0]:+.4f}   y {truth[1]:+.4f}   '
                  f'z {truth[2]:+.4f}')
            print(f'  误差      dx {center[0]-truth[0]:+.4f}   '
                  f'dy {center[1]-truth[1]:+.4f}')
            # z 要跟 center_z 比，不能跟 center[2] 比 ——
            # center[2] 是点云包围盒的中心，而这一簇基本就是**顶面**，
            # 所以 center[2] ≈ 顶面高度（≈0.026），拿它跟物体中心真值（0.0125）
            # 比会凭空报出 +12 mm 的误差，看着像算法坏了，其实比错了对象。
            # 能比的是贴地反推出来的 center_z。
            if center_z is None:
                print('  误差      dz （没算 center_z，跳过）')
            else:
                print(f'  误差      dz {center_z - truth[2]:+.4f}'
                      '   ← 与贴地反推的 center_z 比')

    # 机器可读输出。字段说明（下游 grasp_pipeline 会照这些语义用）：
    #   center_xy   实测，MCD + p1p99 稳健中心。可直接用。
    #   center_z    **反推值**，不是量出来的；相机近乎垂直向下，只拍到顶面。
    #               误差实测 +1.2 mm，别当精确值用。
    #   size_xy     **高估值**。点是物体顶面的投影再叠上边缘与深度噪声，
    #               25 mm 的方块量出 28~29 mm（约 +15%）。要拿它当碰撞盒
    #               尺寸用的话得先减掉这个膨胀，否则 planning 里的方块
    #               比实物大 3~4 mm。
    #   size_z      只反映可见的那层顶面（实测 5.6 mm），不是物体高度。
    #
    # 新增的 validation_* 字段是为了让下游能**看到**判据的余量，而不只是
    # 拿到一个布尔值。之前第 42 条之所以难查，正因为流水线只看到
    # 「感知成功」和一个偏 10 mm 的数字，看不出哪一项判据擦边而过。
    payload = {
        'center_xy': [round(float(center[0]), 5), round(float(center[1]), 5)],
        'center_z': None if center_z is None else round(float(center_z), 5),
        'z_top': None if center_z is None else round(float(z_top), 5),
        'size_xy': [round(float(size[0]), 5), round(float(size[1]), 5)],
        'size_z': round(float(size[2]), 5),
        'z_ground': None if z_ground is None else round(float(z_ground), 5),
        'n_points': int(len(obj)),
        'n_clusters': int(len(clusters)),
        'frame': args.frame,
        # 几何验证的中间量（供日志与标定，勿当精度指标用）
        'validation': {
            'planar_rms_mm': round(vres.planar_rms * 1000, 3),
            'hull_area_ratio': round(vres.hull_area_ratio, 3),
            'normal_score': round(vres.normal_score, 4),
            'mcd_inliers': int(vres.inliers.sum()),
            'residual_rms_mm': round(vres.residual * 1000, 3),
            'checks': {k: {'ok': bool(v[0]), 'value': str(v[1])}
                       for k, v in vres.checks.items()},
        },
    }
    if args.model:
        payload['gazebo_truth'] = truth

    node.destroy_node()
    rclpy.shutdown()
    return finish(0, payload)


if __name__ == '__main__':
    sys.exit(main())
