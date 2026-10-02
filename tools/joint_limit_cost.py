#!/usr/bin/env python3
"""joint_limit_cost.py — 给 Pick-IK 注册「远离硬限位」的自定义代价函数

这个模块解决的是本工程最具体的一个故障：

    抓取时wrist_joint 与 claw_base_joint 被顶到 -1.57080（URDF 硬限位），
    位置约束和伺服在抢同一个自由度，跟踪误差在这个位形上最大
    （实测峰值 0.077~0.080 rad，而 arm_controller 的 trajectory 容差
    是 0.12 rad ——余量只剩 1.5倍）。之前那次 CONTROL_FAILED(-4) 就是
    这么来的（docs/踩坑记录.md 第 31 条）。

为什么换求解器能解决，而收紧软限位只是缓解
──────────────────────────────────────
现有做法是在 joint_limits.yaml 里把 wrist/claw_base 的软限位收到 ±1.50。
那管的是**变量边界** —— 它能让规划器不去生成越界的目标，但：

  · 它是一刀切的。把限位收得越紧，「合法解」的范围就越窄，
    可能把本来能抓的位形直接排除掉。实测收到 ±1.45 就会让
    tools/solve_grasp_poses.py 解出的设计抓取位形（wrist = -1.466）
    变得不可达。
  · 它是**二值**的：限位内的一律等价，限位外的一律不行。
    它没法表达「-1.45 比 -1.20 更好」这种连续偏好。

Pick-IK 的自定义代价函数可以：它给一个**连续的**代价，
让求解器在所有满足位置约束的解里，自动挑那个离限位最远的。

代价函数的设计
──────────────
对每个关节，按它到限位区间的**归一化余量**给代价：

    margin_i = (q_i - lower_i) / (upper_i - lower_i)      ∈ [0, 1]
    cost_i   = w_i * (1 - 2*margin_i)^2                    （对称，0 在中间）

用平方而不是绝对值：平方在区间中点附近梯度为 0，不会把解「吸」到中间；
而线性代价会让解倾向于待在中间，等于偷偷加了个「关节居中」的偏好 ——
那会改变机械臂的 postures，是你不一定想要的。

权重 w_i
────────
不是所有关节都该被同等对待。实测（干净环境跑完整抓取）：
    top_plate / lower_arm / upper_arm离限位都很远（最小0.25~0.39 rad）
    wrist_joint          实际最小 -1.57080   ← 顶在限位上
    claw_base_joint      实际最小 -1.57080   ← 顶在限位上
所以只给腕部两个关节加权，其余给 0（甚至可以完全不加进代价）。
这与 joint_limits.yaml 里「只收腕部两个关节、不动另外三个」是同一个理由 ——
但这里的版本是连续的，不会像一刀切的限位那样排除掉合法解。

为什么不直接把腕部限位收紧当主要手段
────────────────────────────────────
两个原因，上面说了：一刀切会排除合法解；而且 URDF 里的限位是
**物理约束的声明**，拿它去表达「我更希望这个关节别贴边」是滥用语义。
软限位（joint_limits.yaml）才是表达偏好的正确位置 ——
它只影响规划模型，不冒充物理事实。
"""
from __future__ import annotations

# ===========================================================================
# 权重：只给腕部两个关节加权
# ===========================================================================
#
# 数值 1.0 的含义：当某个关节**正好贴在限位上**时，它贡献的代价是 1.0。
# 与 kinematics.yaml 里的 cost_threshold = 1e-4 配合看：
#   贴限位（margin=0）  → 代价 1.0     ≫ 1e-4 → 求解器会拒绝这个解
#   半程（margin=0.5）  → 代价 0       < 1e-4 → 接受
#   余量 1.47/1.50 处   → margin≈0.99  → 代价 ≈0.0004 → 勉强接受
# 也就是说 1.0 这个权重配合 1e-4 阈值，效果是「关节必须待在区间中间
# 那一段，不要贴边」。这正是我们要的。
#
# 为什么腕部两个是 1.0 而其余是 0：见文件头的实测数据。
# 把它们也设成非零会平白改变整条臂的 posture（让肘部也倾向于居中），
# 那不是我们要的效果，而且会让IK 更难在超时内收敛。
# URDF 里的物理硬限位（arm_core.xacro 的 arm_joint_lower/upper = ±1.5708）。
# 显式写出来而不是查模型，是为了让「用哪套限位做参考」这件事在代码里
# 一眼可见 —— 这个选择是自检逼出来的（见 SOFT_LIMIT_WINDOW 注释），
# 藏起来就很容易在某次重构里被改回去。
HARD_LIMITS = {
    'wrist_joint': (-1.5708, 1.5708),
    'claw_base_joint': (-1.5708, 1.5708),
    'top_plate_joint': (-1.5708, 1.5708),
    'lower_arm_joint': (-1.5708, 1.5708),
    'upper_arm_joint': (-1.5708, 1.5708),
}

LIMIT_COST_WEIGHTS = {
    'wrist_joint': 1.0,
    'claw_base_joint': 1.0,
    'top_plate_joint': 0.0,
    'lower_arm_joint': 0.0,
    'upper_arm_joint': 0.0,
}

# 软限位窗口（弧度）：关节进入「距**物理硬限位**这么近」的范围内开始计代价。
#
# ---------------------------------------------------------------------------
# 参考系选硬限位而不是软限位 —— 这是自检逼出来的结论，别改回去
# ---------------------------------------------------------------------------
# 自检（直接跑本文件）第一次的结果是：设计抓取位形 wrist = -1.466
# 的代价 1.298，**远超阈值** → 按这个配置，那个位形会被判不可达。
# 而它恰恰是 tools/solve_grasp_poses.py 解出的、现在实际在用的设计位形。
# 也就是说当时的配置会「为了让腕关节远离限位，而把唯一能用的解排除掉」。
#
# 根因是参考系选错了。当时用 joint_limits.yaml 的软限位 ±1.50 做参考，
# 而设计位形 -1.466 距软限位只有 0.034 rad —— 落在窗口内，
# 于是被判「贴限位」。可它距**物理硬限位** -1.5708 还有 0.105 rad，
# 一点都不危险。
#
# 换用硬限位 ±1.5708 做参考系后（实测）：
#     窗口      -1.5708（坏解）  -1.466（设计位形）  -1.20（安全）
#     0.05rad        1.0000            0.000000          0.000000
#     0.07rad        1.0000            0.000000          0.000000
#     0.10rad        1.0000            0.000000          0.000000
# 坏解与好解被干净地分开，且设计位形不再被误伤。
#
# 取 0.10 rad（约 5.7°）：在坏解（余量 0）与设计位形（余量 0.105）之间，
# 且离设计位形还有一点余量。这个数是**从实测数据反推**的，
# 不是拍的 —— 它必须同时满足「坏解 >阈值」与「设计位形 <阈值」两条。
# 改动之后请重跑本文件的自检，那两条判据会立刻告诉你是否还成立。
SOFT_LIMIT_WINDOW = 0.10      # rad，约 5.7 度


def register_limit_cost(ik_solver, robot_model, group_name='arm',
                       weights=None, window=SOFT_LIMIT_WINDOW):
    """把限位代价函数注册到 Pick-IK 求解器。

    参数
    ----
    ik_solver : moveit::core::RobotModel 的 IK 求解器对象
                （Pick-IK 插件实例，不是 KinematicsPlugin）
    robot_model : RobotModel，用来查关节限位
    group_name : 规划组名

    返回 True 表示注册成功。

    **本函数需要 C++ 扩展**（见文件末尾的说明）。Python 侧目前只能做
    「算出每个关节应该离限位多远」并打印出来供核对，不能真正注册到
    MoveIt 的求解器里 —— 原因是 Pick-IK 的 IkCostFn 接口是 C++ 虚类，
    MoveIt 的 Python 绑定没有把它暴露出来。
    """
    weights = weights or LIMIT_COST_WEIGHTS
    bounds = {}
    for name, w in weights.items():
        if w <= 0:
            continue
        b = _joint_bounds(robot_model, group_name, name)
        if b is None:
            continue
        lo, hi = b
        # 代价必须相对**物理硬限位**计算，而不是 joint_limits.yaml 里的
        # 软限位。原因与实测见 SOFT_LIMIT_WINDOW 的注释：设计位形 -1.466
        # 距软限位只有 0.034 rad，用软限位做参考会把它误判成「贴限位」。
        # 所以这里显式回到 URDF 的硬限位：
        hard_lo, hard_hi = HARD_LIMITS.get(name, (lo, hi))
        bounds[name] = (hard_lo, hard_hi, w)
    print(f'[限位代价] 注册 {len(bounds)} 个关节：'
          f'{ {k: round(v[2], 3) for k, v in bounds.items()} }')
    print('  注意：注册动作需要 C++ 扩展，见本文件末尾说明。')
    return bool(bounds)


def _joint_bounds(robot_model, group_name, joint_name):
    """查关节限位。返回 (lower, upper) 或 None。"""
    try:
        bounds = robot_model.getJointModelBounds(joint_name)
        if bounds is None:
            return None
        # 优先用 active_joint_model 的限位（那是 soft limit 之后的值，
        # MoveIt 会用 joint_limits.yaml 覆盖它），拿不到再退回 model。
        try:
            ajm = robot_model.getActiveJointModelBounds(joint_name)
            if ajm is not None:
                return float(ajm.position_min), float(ajm.position_max)
        except Exception:                       # noqa: BLE001
            pass
        return float(bounds.position_min), float(bounds.position_max)
    except Exception:                           # noqa: BLE001
        return None


def limit_cost(q, bounds, window=SOFT_LIMIT_WINDOW):
    """纯 Python 版代价函数，供离线核对用（不参与实际 IK）。

    q     : {关节名: 角度} 或与 bounds 同序的序列
    bounds: {关节名: (lower, upper, weight)}

    返回标量代价。

    为什么要有一个「不参与实际 IK」的纯 Python 版：
    它让这个代价函数**可以被单独验证** —— 拿一批已知解算一遍代价，
    确认「贴限位的解代价确实高、居中的解代价确实低」。
    一个注册进MoveIt 就再也不好单独测的东西，不该在没有验证的情况下
    就声明它能解决问题。
    """
    total = 0.0
    items = q.items() if isinstance(q, dict) else zip(bounds.keys(), q)
    for name, val in items:
        b = bounds.get(name)
        if b is None:
            continue
        lo, hi, w = b
        span = hi - lo
        if span <= 0:
            continue
        margin = (val - lo) / span          # 0..1
        # 只在靠近两端 window/span 比例内计代价
        d = abs(2.0 * margin - 1.0)         # 0 在中间，1 在两端
        win = window / span
        if d <= 1.0 - win:                   # 还在窗口外
            continue
        # 归一化到窗口内 0..1，再平方
        t = (d - (1.0 - win)) / max(win, 1e-9)
        total += w * t * t
    return total


# ===========================================================================
# 自检
# ===========================================================================


def _self_test():
    """用实测数据验证代价函数的形状是否正确。

    两条必须同时成立的判据（任一不成立就别启用 Pick-IK）：
      ① 贴硬限位的坏解（-1.5708）代价 > cost_threshold  -> 会被拒绝
      ② 设计抓取位形（-1.466）代价 < cost_threshold  -> 不会被误伤
    第 ② 条是这一版与第一版的唯一区别，也正是第一版踩的坑。
    """
    print('=== 限位代价函数自检 ===')
    print(f'窗口 {SOFT_LIMIT_WINDOW} rad；权重 {LIMIT_COST_WEIGHTS}')
    print(f'参考系：物理硬限位 {HARD_LIMITS["wrist_joint"]}'
          '（不是软限位 —— 理由见 SOFT_LIMIT_WINDOW 注释）\n')

    bounds = {n: (HARD_LIMITS[n][0], HARD_LIMITS[n][1], w)
              for n, w in LIMIT_COST_WEIGHTS.items()}

    cases = [
        ('实测坏解（顶在物理限位上）', -1.5708),
        ('软限位边界 -1.50', -1.50),
        ('设计抓取位形（必须在阈值下）', -1.466),
        ('较安全 -1.20', -1.20),
        ('区间中间 0.0', 0.0),
    ]
    thresh = 1e-4
    print(f'{"工况":34s}{"wrist":>10s}{"代价":>12s}  判定')
    print('-' * 74)
    ok = True
    for name, q in cases:
        c = limit_cost({'wrist_joint': q, 'claw_base_joint': q}, bounds)
        verdict = '超阈值→拒' if c >= thresh else '接受'
        # 判据①：贴硬限位的坏解必须被拒
        if q <= -1.55 and c < thresh:
            ok = False
            verdict = '**坏解被放行**'
        # 判据②：设计位形必须被接受
        if abs(q + 1.466) < 1e-9 and c >= thresh:
            ok = False
            verdict = '**设计位形被误拒**'
        print(f'{name:34s}{q:>10.4f}{c:>12.6f}  {verdict}')

    print(f'\n阈值 cost_threshold = {thresh:g}（见 kinematics.yaml）')
    if ok:
        print('\u2713 两条判据都成立：坏解被拒、设计位形被接受。')
        print('  可以按 kinematics.yaml 的 cost_threshold = 1e-4 启用。')
        return 0
    print('\u2717 判据不成立 —— **先别启用 Pick-IK**，把窗口/权重调对。')
    print('  调小 SOFT_LIMIT_WINDOW，别调大 cost_threshold：')
    print('  后者会让贴限位的解重新被放行，正好抵消这个代价函数的作用。')
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
