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

分割思路：
  视野里绝大多数点是地面，所以先取 z 的直方图峰值当地面高度，把地面附近
  的点剔掉，剩下的做体素连通域聚类，取最大的一簇当目标。
  纯 numpy + scipy.ndimage 实现，不依赖 PCL / open3d（本机没装）。

  注意点云是在相机系里的，单位是米，但相机装在腕上会随臂运动，
  所以先经 TF 转到 base_link 再做几何判断。真值对照也在 base_link 下做，
  两边参考系必须一致，否则误差里混进的是坐标系差异而不是算法误差。
"""
import argparse
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
    args = ap.parse_args()

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
        return 2
    if 'msg' not in joints:
        print('超时：没收到 /joint_states，无法按 URDF 剔除自身遮挡。',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 2
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
        return 3
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
        return 4
    print(f'  聚成 {len(clusters)} 簇，取最大的一簇')

    obj = clusters[0]
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    center = (lo + hi) / 2.0
    size = hi - lo

    print(f'\n=== 目标位姿（{args.frame} 系）===')
    print(f'  点数    {len(obj)}')
    print(f'  中心 xy x {center[0]:+.4f}   y {center[1]:+.4f}')
    print(f'  观测范围 dx {size[0]:.4f}  dy {size[1]:.4f}  dz {size[2]:.4f}')
    print(f'  z 区间   [{lo[2]:+.4f}, {hi[2]:+.4f}]')

    # 相机几乎垂直向下（实测光轴与竖直只差 8.5 度），25 mm 高的侧面在图像里
    # 只摊开 25*sin(12.3) ≈ 5 mm —— 于是点云实际只覆盖了目标顶面，中心高度
    # 根本测不到。只能靠「物体贴地」这个前提反推：平顶物体贴地放置时，
    # 顶面离地高度就等于它的高度。这条臂本来也够不着台面上的东西
    # （见 docs/使用说明书.md 第七节），所以这个前提在本工程里始终成立。
    if args.frame == 'base_link' and z_ground is not None:
        z_top = float(hi[2])
        print(f'  顶面高度 {z_top:+.4f}   离地高度 {(z_top - z_ground)*1000:.1f} mm')
        print(f'  贴地反推的中心 z = {(z_ground + z_top) / 2:+.4f}'
              '（相机几乎垂直向下，测不到中心高度）')

    if args.model:
        truth = gazebo_truth(args.model)
        print(f'\n=== 与 Gazebo 真值对照（{args.model}）===')
        if truth is None:
            print('  读不到真值（ign 命令不可用或模型名不对）')
        else:
            print(f'  真值中心  x {truth[0]:+.4f}   y {truth[1]:+.4f}   '
                  f'z {truth[2]:+.4f}')
            print(f'  误差      dx {center[0]-truth[0]:+.4f}   '
                  f'dy {center[1]-truth[1]:+.4f}   '
                  f'dz {center[2]-truth[2]:+.4f}')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
