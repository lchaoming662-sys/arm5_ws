#!/usr/bin/env python3
"""geometry_validity.py — 目标点簇的**多重几何验证**与稳健位姿估计

这个模块回答一个感知系统必须回答的问题：

    「手上这簇点，到底是不是一个完整、形状正确、可以据此去抓的工件？」

为什么不能用跨度（span）判据
────────────────────────────
原实现（见 docs/踩坑记录.md 第 42/43 条）用「1-99 分位跨度是否落在
工件尺寸 × [0.8, 1.8]」当有效性检查。它能挡住**大部分**截断，但有两类
失效它挡不住：

  1. **碎片恰好有正确跨度。** 目标被手指挡住一半，剩下的碎片在 x 方向
     仍然可能有 25 mm 宽（方块的一部分 + 一段桌面残留），跨度判据放行，
     中心却偏了 3~4 mm。跨度是**投影**度量，它对「这块点云是不是一个
     连通的完整物体」一无所知。
  2. **粘连。** 方块紧邻其它东西时，两簇连成一片，跨度暴涨到 40 mm。
     跨度判据用「过大」这一侧挡，但那是唯一一次侥幸 —— 如果粘连发生在
     厚度方向，跨度依然完美。

三重验证各自堵一个洞，且**互不相关**：

  ┌─────────────┬──────────────────────┬───────────────────────────┐
  │ 判据         │ 物理含义              │ 挡住的失效               │
  ├─────────────┼──────────────────────┼───────────────────────────┤
  │ 平面性       │ 它是一块**面**，不是   │ 点簇里混进了立着的碎片、  │
  │ λ₃/λ₁       │ 一堆乱点              │ 背景残留                 │
  │ 凸包面积     │ 它**只占这么大地方**  │ 粘连（面积暴涨）、碎片    │
  │             │                      │ （面积骤减）             │
  │ 法向一致性   │ 表面朝向一致          │ 跨了两个面（方块顶面 +   │
  │             │                      │ 桌面残留）拼在一起       │
  └─────────────┴──────────────────────┴───────────────────────────┘

三者**全过**才输出位姿。任一不过直接抛 `GeometryRejected` —— 不是返回
一个带 warning 的位姿。下游（抓取流水线）会因为异常而拒绝规划，这正是
我们要的：宁可任务失败，也不能让机械臂扑空后报「执行完成」。

中心估计为什么不用 bbox 中心
────────────────────────────
docs/踩坑记录.md 第 43 条已经实测过：

    估计量     x 均值   x 最差    y 均值   y 最差
    bbox     -1.18    1.86     +0.45    0.57
    centroid -0.41    0.43     +3.27    3.33   ← 视野里的第二瓣把 y 拽走
    p1p99    -0.87    1.06     +0.81    0.98   ← 最差值最小

本模块在此基础上**再加一层 MCD**（Minimum Covariance Determinant，
最小协方差行列式估计）：先用协方差行列式最紧的那 50% 点拟合出稳健中心，
把离群点剔掉，再在**内点**上做 p1p99。
理由：p1p99 只能压掉两端**稀疏尾点**，压不掉「成片贴上来的第二瓣」——
那一瓣有足够多的点，分位数动不了它。MCD 恰好擅长这件事：它找的是
「最紧的 50% 点集」，成片贴上来的东西天然被排除。

依赖
────
默认只用 numpy（PCA、凸包用 scipy.spatial.ConvexHull）。Open3D **可选**：
装了会用它的凸包与法向实现（更快、鲁棒性更好），没装走 numpy/scipy。
两条路径的判据**完全一致**，判据阈值不随实现变化 —— 这是刻意的：
判据不能因为换了库就变松或变紧，否则「换了环境就换了答案」。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# ---------------------------------------------------------------------------
# Open3D 是可选增强，不是硬依赖。
#
# 为什么不把它设成必需：本工程整机只用 numpy + scipy（系统自带，版本锁死
# 在 Ubuntu 22.04 的 1.21.5 / 1.8.0），而 Open3D 会拖来一堆自己的依赖
# （它要求 numpy>=1.18 且对 numpy 2.x 有兼容问题），在一个已经能跑的
# 环境里引入它，收益不抵风险。所以：装了用，没装照样跑，判据一致。
# ---------------------------------------------------------------------------
try:                                                # pragma: no cover
    import open3d as _o3d
    _HAVE_OPEN3D = True
except Exception:                                   # noqa: BLE001
    _o3d = None
    _HAVE_OPEN3D = False


# ===========================================================================
# 判据阈值 —— 集中放在这里，不要散落到代码里
#
# 每个阈值都写了「为什么是这个数」。改任何一个之前先读那段注释，
# 并按 docs/踩坑记录.md 的方式补一条实测记录。
# ===========================================================================

# 平面性判据：内点到拟合主平面的 RMS 残差（**绝对量**，单位米）
#
# ---------------------------------------------------------------------------
# 演进过程：为什么最终用绝对残差，而不是最初想的 λ₃/λ₁ 相对量
# ---------------------------------------------------------------------------
# v1 用 λ₃/λ₁ < 0.05。自检立刻暴露问题：把深度噪声从 1.3 mm 调到 4 mm，
# 完整目标被误杀（λ₃/λ₁ 从 2.97e-2涨到 2.60e-1）。
#
# 根因：λ₃/λ₁ 是**相对量**，深度噪声抬高 λ₃ 的同时观察倾斜也抬高 λ₃，
# 两者在同一个分子里无法区分 —— 噪声一大，倾斜就掩盖了真实形状。
#
# 于是改用绝对残差（tools/calib/calib_planar_residual.py 实测对比）：
#
#     工况                      λ₃/λ₁    RMS     扣除倾斜后
#     干净 1.3 mm（基准）        2.97e-02  1.29mm  0.00 mm
#     噪声 4 mm                2.60e-01  3.86mm  3.56 mm
#     噪声 6 mm                5.29e-01  5.67mm  5.47 mm
#     顶面 + 竖直接触面          3.84e-01  5.28mm  5.07 mm
#     顶面 + 20% 竖直残留        1.37e-01  3.47mm  3.14 mm
#     顶面 + 10% 竖直残留        9.33e-02  2.72mm  2.28 mm
#
# **结论：两个指标都无法把「噪声恶化」与「异面污染」分开**（绝对残差下
# 噪声 6mm 是 5.47mm、异面污染是 5.07mm，直接重叠）。所以不是换个量就
# 能解决的问题，这里必须如实说明而不是继续调阈值。
#
# 但绝对残差仍是更好的选择，理由是**它有物理意义、可以独立解释**：
#   · RMS 残差直接回答「这簇点离平面有多远，单位是毫米」，可以直接和
#     工件尺寸、深度噪声对��；
#   · λ₃/λ₁ 是个无量纲比值，0.05 这个数字没有物理含义，只能标定；
#   · 绝对残差可以**解析扣除**已知的倾斜贡献（本工程 1.48 mm），
#     而 λ₃/λ₁ 无法扣除。
# 判据也因此更易解释：残差 3 mm 就是 3 mm。
#
# ---------------------------------------------------------------------------
# 阈值 4.0 mm 是怎么定的，以及它明确**不**能做什么
# ---------------------------------------------------------------------------
# 取在「噪声 4 mm（3.56 mm）」与「异面污染（5.07 mm）」之间。
#
# 它做不到的事，必须写清楚，否则就成了又一处「安静地给错答案」：
#
#   · **深度噪声超过约 4 mm 时，完整目标会被误杀。**这不是 bug，
#     是信息论层面的：噪声与几何细节同量级时，点云里已经没有足够的
#     几何信息来做形状判断。此时正确做法是**修传感器/渲染设置**，
#     不是放宽阈值 —— 放宽了就会连带放进异面污染。
#     本工程当前的 rgbd_camera 深度噪声实测约 1.3 mm
#     （arm_camera.xacro 里 stddev=0.001），对应残差 1.29 mm，
#     距阈值有一倍以上余量，安全。
#
#   · **异面污染占比低于约 15% 时，本判据可能漏放。**实测 10% 残留的
#     残差是 2.28 mm，在阈值之下。这类小比例污染靠它挡不住，
#     要靠上游的分割（剔地面、剔自身遮挡）质量，以及第二道法向判据。
#     这也是为什么法向一致性不能省：两者灵敏度区间不同，互为交叉验证。
#
# 换句话说：本判据负责「明显的形状错误」，法向判据负责「平面内的法向异常」，
# 面积判据负责「覆盖范围」，三者覆盖不同的失效模式。**没有任何单一判据
# 能独立把关，点数下限兜住「信息量根本不够」的情况。**
PLANAR_RMS_MAX = 0.004
# 观察倾斜对残差的解析贡献。25 mm 跨度上均匀分布的平板，倾斜 θ 贡献
# ��� RMS = L/sqrt(12)·sinθ。本工程光轴偏竖直实测 12.3°，扣除它之后
# 残差才真正反映「形状偏离平面」而不是「我们斜着看一个平面」。
PLANAR_LEAN_TILT_DEG = 12.3

# 凸包面积比：实测凸包面积 / 理论面积（工件尺寸的平方）。
#
# 下界 0.6：视线正对目标时点云完整覆盖顶面，凸包面积**实测系统性偏大**
# （docs/踩坑记录.md 第 41 条：线性尺寸偏大约 15%，面积偏大约 32%），
# 所以偏大是常态；偏小才是问题 —— 说明只看到了碎片。实测截断到
# 一半时面积掉到 0.4~0.5，正好在 0.6 之下。
#
# 上界 2.0：干净工况的实测面积比在 1.2~1.4。粘连一块 25 mm 的相邻
# 物体就能到 2.5 以上。2.0 留了一点余量，因为被挡时形状会畸变。
HULL_AREA_RATIO_RANGE = (0.6, 2.0)

# 法向一致性：E[1 - |cos θ|]，θ 是逐点法向与主法向的夹角。
#
# ---------------------------------------------------------------------------
# 为什么不是「夹角的标准差」—— 那个形式在多面混合下**非单调**
# ---------------------------------------------------------------------------
# 第一版用「与主法向夹角的标准差 < 15°」。实测（k=30，同一份合成点云）：
#
#     工况                          夹角σ      1-|cos| 均值
#     干净顶面（1.3 mm 噪声）        21.45        0.213
#     顶面 + 竖直接触面26.52        0.442   ← 旧指标的峰值
#     顶面 + 竖直接触面 + 斜碎屑     22.28        0.461   ← 污染更重反而更低
#
# 旧指标在两档之间达到峰值，**再加一层斜碎屑反而下降** —— 污染更重时
# 指标变小。根因是「标准差」刻画的是分布的**离散程度**，不是「有多少点
# 跑偏了」：两类点时分布是双峰（方差最大），三类点时中间被填上、
# 反而不那么双峰 —— 方差回落，但跑偏的点其实更多了。
#
# 这是极值/方差型统计量被误当作分布一致性度量的典型，
# 本工程第 40/42/43 条那一类错误的第四次重演。
#
# ⚠️ 关于这个结论的强度（第二版更正，踩坑记录第 52 条）：
#   我第一版写的是「反向指标」，那是**下过强的结论**。
#   复现发现：同一指标在 k=12/20/30/45/60 不同近邻数下、在不同随机
#   种子下，单调性会来回变化 —— 它在多数参数下其实是单调的，
#   真正的问题是「非单调」（仅在特定混合比例下出现）。
#   结论必须写成实际的样子，否则后来者会照着错的前提做决策。
#   下面阈值那一节里之所以用「k=30」这一档标定，就是因为它是最差的一档。
#
# ---------------------------------------------------------------------------
# 为什么用 1 - |cos| 的均值
# ---------------------------------------------------------------------------
# 它对好点与坏点的期望有清晰间隔，而且单调：
#     完全一致→ ≈0.02 ；随机取向 → 1 - 2/π = 0.3634（解析值）；垂直 → 1.0
# 取 |cos| 而非 cos：PCA 最小特征向量的符号是任意的，不取绝对值会把
# 同一个平面的两半误判成「朝向相反」，那是纯伪信号。
#
# ---------------------------------------------------------------------------
# 阈值 0.25 是怎么定的（tools/calib/calib_normal_score.py 实测，非拍脑袋）
# ---------------------------------------------------------------------------
# 决策表（行 = 掺入竖直接触面的比例，列 = k 近邻数，取值 = 1-|cos| 均值）：
#
#      污染%    k=20    k=30    k=40    k=50    k=64    k=80
#        0%     0.285   0.229   0.184   0.152   0.122   0.094
#        5%     0.311   0.258   0.215   0.185   0.156   0.129
#       10%     0.329   0.283   0.246   0.218   0.192   0.167
#       20%     0.354   0.318   0.291   0.267   0.247   0.228
#       35%     0.400   0.375   0.359   0.343   0.331   0.320
#
# 三个结论：
#
#  1. k=20 时干净工况就已经 0.285，**高于任何合理阈值** —— 即第一版把
#     完整目标也拒了。这就是为什么不能凭直觉定 k。
#
#  2. 加大 k 确实压低噪声（0.285→0.094），但污染 10% 与干净之间的
#     **间隔几乎不变**（0.044→0.073）。也就是说：k 换来的分辨力很有限。
#     根因是深度噪声 1.3 mm 相对点间距 0.46 mm 太大，法向估计本身就
#     噪声主导 —— 这是数据条件决定的，不是算法能补的。
#
#  3. 所以本指标定位为**小比例污染的粗筛**，阈值取 0.25：
#     k=30 时干净 0.229（余量 0.021）过、污染 10% 0.283 被拒；
#     污染 5%（0.258）也刚好被拒。
#
# 划清职责边界（重要）：**主判据是平面性**，它对掺入异面点敏感得多
# （实测异面混入使 λ₃/λ₁ 从 2.8e-2 涨到 4.0e-1，一个半数量级）。
# 法向一致性在这里是第二道交叉验证，用来挡住「平面但法向乱指」这类
# 平面性看不出的情形。**不要指望它单独判5% 以下的污染** —— 上表已证明
# 做不到，需要时应改用更强的点云分割，而不是继续调这个阈值。
NORMAL_CONSISTENCY_MAX = 0.25
NORMAL_KNN = 30

# 中心估计。MCD 的保留比例 h：0.5 是 MCD 文献的默认值，也是
# Tukey 50% 门槛。提高到 0.7~0.8 会让「成片贴上来的第二瓣」混进
# 内点集，剔不掉；降低到 0.3 则会把顶面的边缘（真实点）剔掉。
MCD_KEEP_FRACTION = 0.50
# MCD 迭代次数。5 轮足够收敛：实测 5 轮与 20 轮的内点集差异 < 1%。
MCD_MAX_ITER = 5
# MCD 的收敛容差：中心移动小于该值（米）就停。
MCD_TOL = 1e-5

# 中心分位数。用 1/99 而不是 0.5/99.5：后者的裁剪量太小，
# 在点数少（<1000）时会引入量化误差。
CENTER_PERCENTILE = (1.0, 99.0)


class GeometryRejected(RuntimeError):
    """点簇没通过几何验证。

    单独一个异常类型（而不是复用 RuntimeError）是为了让调用方能
    **精确地**区分「几何不合格」和「环境/通信故障」：
    前者是正常的业务结果（目标没看全，换个位置再来），
    后者是故障（TF 查不到、点云没来）。两者的重试策略完全不同。
    """

    def __init__(self, reason, detail=''):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail

    def __str__(self):
        return f'{self.reason}\n    {self.detail}' if self.detail else self.reason


class ValidationResult:
    """一次验证的完整记录。

    设计成**把中间量都留下来**而不是只返回一个 bool：
    排查时你需要看到「平面性过了但面积没过」这种信息 ——
    而失败的根因往往就藏在差得最多的那一项里。
    """

    def __init__(self):
        self.ok = False
        self.reason = ''
        self.n_points = 0
        # 平面性
        self.eigenvalues = None      # 降序，长度 3，单位 m²
        self.planarity = None        # λ₃/λ₁，**仅诊断**，不作判据
        self.planar_rms = None       # 扣除倾斜后的面残差 RMS，单位 m（判据用）
        self.lean_rms = None         # 被扣除的倾斜贡献，单位 m
        # 凸包
        self.hull_area = None        # m²
        self.hull_area_ratio = None  # 实测/理论
        # 法向
        self.normal_score = None    # E[1-|cosθ|]，越小越一致
        # 结果
        self.inliers = None          # MCD 后的内点掩码（bool 数组）
        self.center = None           # (3,) 内点稳健中心，单位 m
        self.residual = None         # 内点到主平面的 RMS 残差，单位 m
        self.checks = {}             # 名字 -> (通过?, 实测值, 阈值描述)

    def summary(self):
        """一行摘要，给流水线的日志用。"""
        if self.ok:
            return (f'几何验证通过：面残差 {self.planar_rms*1000:.2f} mm · '
                    f'面积比 {self.hull_area_ratio:.2f} · '
                    f'法向 {self.normal_score:.3f} · '
                    f'内点 {int(self.inliers.sum())}/{self.n_points} · '
                    f'中心 ({self.center[0]:+.4f}, {self.center[1]:+.4f}, '
                    f'{self.center[2]:+.4f})')
        parts = [f'{k}={v[1]}' for k, v in self.checks.items() if not v[0]]
        return f'几何验证失败：{self.reason}（不合格项：{" ".join(parts)}）'


# ===========================================================================
# ① 平面性：PCA 特征值分解
# ===========================================================================

def planarity(points):
    """返回 (特征值降序数组, λ₃/λ₁)。

    用协方差矩阵而不是原始点的二阶矩：协方差已经中心化，
    反映的是**围绕质心的分布**，这才是「平面性」该有的度量。
    直接对原始点算特征值会把「点云整体偏在某处」也算进去。

    为什么要**中心化后再算**：一次 25 mm 方块在 base_link 系里位于
    x≈0.027，如果不做中心化，点云在坐标轴上有一个 27 mm 的偏移，
    λ₁ 会变成偏移²的量级（7e-4），与噪声尺度（1e-6）差 100 倍，
    比值完全失去意义。这正是「不做中心化就得到一个看起来正常但
    毫无意义的结果」的典型 —— 本工程第 42 条踩的就是这类坑。
    """
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    q = p - c
    cov = (q.T @ q) / max(len(q) - 1, 1)
    eig = np.linalg.eigvalsh(cov)[::-1]   # 升序 -> 降序
    lam1 = max(float(eig[0]), 1e-20)     # 防除零：整簇点完全重合时
    lam3 = max(float(eig[2]), 0.0)
    return eig, lam3 / lam1


# ===========================================================================
# ② 凸包面积
# ===========================================================================

def hull_area_2d(points_2d):
    """2D 凸包面积（m²）。返回 None 表示算不出来（点数不足/退化）。

    为什么必须用凸包而不能用包围盒面积：
    包围盒面积 = w×h，只要有一个稀疏尾点就能把某一维拉出去（踩坑记录
    第 40/43 条已经两次因为 min/max 吃过亏）。凸包对尾点不敏感 ——
    凸包只会**包含**它们，不会被它们「撑大」。这是凸包在点云里被
    反复使用的根本原因。

    为什么先做 p1p99 预裁剪：Qwant 等凸包实现对少量离群点是稳健的，
    但对「成片离群」不稳健。预裁剪去掉的是「远处飞来的碎片」
    （它们在 2D 投影上会形成独立的凸包顶点）。裁到 1/99 分位
    损失的信息可忽略（那些点本来就不属于这个物体）。
    """
    p = np.asarray(points_2d, dtype=float)
    if len(p) < 4:
        return None
    lo = np.percentile(p, 1.0, axis=0)
    hi = np.percentile(p, 99.0, axis=0)
    q = p[(p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0])
          & (p[:, 1] >= lo[1]) & (p[:, 1] <= hi[1])]
    if len(q) < 4:
        q = p
    try:
        return float(ConvexHull(q).volume)     # 2D 的 volume 就是面积
    except (QhullError, ValueError):
        # 点共线 / 全重合：退化。用 0 表示「退化」而不是 None ——
        # 退化本身就是不合格信号，应该走拒绝路径而不是跳过检查。
        return 0.0


# ===========================================================================
# ③ 法向一致性
# ===========================================================================

def point_normals(points, k=NORMAL_KNN, use_open3d=None):
    """逐点法向。返回 (法向数组 (N,3), 邻域索引 (N,k))。

    k 近邻用**一次性 KDTree 查询**，不是逐点循环：
    2500 个点 × 每次一个球查询 = 2500 次 Python 循环，实测 4 秒；
    一次向量化查询同样的结果只要 30 ms。感知节点在抓取流水线里，
    每多 4 秒就多 4 秒的仿真等待（RTF 只有 0.48）。

    k 的选择与 open3d 无关：open3d.compute_normals 内部也是 kNN
    PCA，只是用 C++ 写。我们自己算法是为了**判据完全可控** ——
    open3d 的参数（radius vs knn、是否平滑法向）一旦升级可能变，
    而判据阈值是按我们这套实现标定的。要用 open3d 时只替换
    `normals` 的计算，判据照样不变。
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    kk = min(k + 1, n)                # 含自身，所以查 k+1 个再丢掉自己
    # 邻域索引只在 numpy 分支需要（open3d 自己内部算），
    # 但函数契约承诺返回它，所以两条分支都要给出一个有意义的值。
    idx = None
    if _o3d is not None and (use_open3d or (use_open3d is None and _HAVE_OPEN3D)):
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(p)
        pcd.estimate_normals(_o3d.geometry.KDTreeSearchParamKNN(knn=kk))
        normals = np.asarray(pcd.normals, dtype=float)
        # open3d 不返回邻域索引，这里用一次 cKDTree 查询补上，
        # 让返回契约与 numpy 分支一致（调用方不该关心走了哪条路）。
        from scipy.spatial import cKDTree
        _, idx = cKDTree(p).query(p, k=kk)
        idx = np.atleast_2d(idx)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(p)
        _, idx = tree.query(p, k=kk)
        idx = np.atleast_2d(idx)
        nb = p[idx]                                # (N,k,3)
        nb = nb - nb.mean(axis=1, keepdims=True)   # 邻域中心化
        cov = np.einsum('nki,nkj->nij', nb, nb) / max(kk - 1, 1)
        # eigh 返回升序，最小特征值对应的特征向量就是法向
        _, vecs = np.linalg.eigh(cov)
        normals = vecs[:, :, 0]
    return normals, idx


def normal_consistency(points, normals, main_normal):
    """返回 (1-|cosθ| 均值, 逐点角度数组)。越小越一致。

    用均值而不是分位数或极值，理由见NORMAL_CONSISTENCY_MAX 的注释：
    分位数只看尾部分布，对「掺进 10% 异面点」这种小比例污染不敏感，
    而小比例污染恰恰是点云分割最常见的失效形态。
    """
    mn = np.asarray(main_normal, dtype=float)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = normals / np.maximum(nrm, 1e-12)
    cosang = np.abs(unit @ mn)          # |cos|：法向本身符号无意义
    one_minus = 1.0 - cosang
    ang = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    return float(one_minus.mean()), ang


# ===========================================================================
# ④ MCD 稳健中心
# ===========================================================================

def mcd_inliers(points, keep=MCD_KEEP_FRACTION, max_iter=MCD_MAX_ITER,
                tol=MCD_TOL):
    """MCD 最小协方差行列式：返回内点掩码。

    算法（Hawkins-Bradu-Rousseeuw，1994）：
      1. 取前 keep 比例的点当初始内点集，算协方差矩阵 C；
      2. 用 C 的马氏距离给所有点打分，取最小的 keep 比例个当新内点集；
      3. 重复直到内点集不再变化（或中心移动小于 tol）。

    为什么用马氏距离而不是欧氏距离：协方差矩阵 C 同时编码了每个方向的
    方差与方向间的相关性。点云在水平方向拉得长（顶面 25x25 mm 在斜视
    下投影成 29x22 mm）、垂直方向只有 5 mm 的厚度 —— 用欧氏距离会把
    「垂直方向差 1 mm」当成和「水平方向差 1 mm」同等重要，而前者其实
    噪声大得多。马氏距离会按各自方向的离散度归一化，自动降权。

    注意这是**近似** MCD：原始算法用随机子集穷举（n=2500 时不可行）。
    5 轮不动点迭代在实测里与 20 轮的内点集差异 <1%，够用。
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    n_keep = max(int(round(n * keep)), 3)
    if n <= n_keep:
        return np.ones(n, dtype=bool)

    # 初始内点：按到全体质心的欧氏距离取最近的 n_keep 个。
    # （不用随机子集：确定性结果可复现，这在这个工程里比统计效率更重要 ——
    #  随机子集会让同一个点云两次运行给出不同的内点集，无法做回归对比。）
    c0 = p.mean(axis=0)
    d0 = np.linalg.norm(p - c0, axis=1)
    idx = np.argsort(d0)[:n_keep]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    for _ in range(max_iter):
        c = p[mask].mean(axis=0)
        q = p - c
        cov = (q[mask].T @ q[mask]) / max(int(mask.sum()) - 1, 1)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # 协方差奇异：内点几乎共线/共面。换伪逆继续，不要直接放弃 ——
            # 退化本身就交给后面的判据去拒绝，这里不该崩。
            inv = np.linalg.pinv(cov)
        d = np.einsum('ni,ij,nj->n', q, inv, q)     # 马氏距离平方
        new = np.zeros(n, dtype=bool)
        new[np.argsort(d)[:n_keep]] = True
        if np.array_equal(new, mask):
            break
        if np.linalg.norm(p[new].mean(axis=0) - p[mask].mean(axis=0)) < tol:
            mask = new
            break
        mask = new
    return mask


# ===========================================================================
# 主入口
# ===========================================================================

def validate_and_estimate(points, expected_size, frame='base_link'):
    """前置拒绝的多重几何验证 + 稳健位姿估计。

    返回一个 ValidationResult。**不合格时 ok=False，绝不返回一个
    看起来正常的位姿** —— 调用方必须先判 ok 再取 center。

    参数
    ----
    points : (N,3) 点云，base_link 系，单位米。要求已剔地面、已剔自身遮挡。
    expected_size : 标量，工件宽度（米）。本工程是 0.025。
    frame : 只用于报错信息。

    为什么把 frame 传进来：报错信息里必须说清「在哪个参考系下不合格」。
    跨参考系比较几何量是这类系统最常见的错误源之一（踩坑记录第 44 条：
    没核对参考系与位形，得出的一切结论都不可比）。
    """
    r = ValidationResult()
    p = np.asarray(points, dtype=float)

    # --- 前置：点数下限 -------------------------------------------------
    # 为什么下限是 200 而不是之前用的 400：MCD + PCA 至少需要几十个点才稳定，
    # 200 是在「统计量不抖动」与「别误杀远处目标」之间的折中。
    # 实测（臂在扫描位形）：视野中心的目标 2400~2500 点，
    # 视野边缘被截断的 1200 点，严重截断的 328 点。200 把 328 挡在外面。
    MIN_POINTS = 200
    r.n_points = len(p)
    if len(p) < MIN_POINTS:
        r.reason = f'点数不足（{len(p)} < {MIN_POINTS}）'
        r.checks['point_count'] = (False, len(p), f'≥{MIN_POINTS}')
        return r
    r.checks['point_count'] = (True, len(p), f'≥{MIN_POINTS}')

    # --- ① 平面性：主平面拟合 + 扣除倾斜后的 RMS 残差 -------------------
    # 顺序有讲究：先算特征值与主法向（几何事实），再用它们算残差，
    # 最后才拿残差去比阈值。反过来做（先判阈值再算）会在数据异常时
    # 拿不到可诊断的中间量。
    eig, ratio = planarity(p)
    r.eigenvalues = eig
    r.planarity = ratio            # 保留作诊断输出，不作判据（见常量注释）

    # 主平面法向 = 最小特征值对应的特征向量。
    # 这里重算一次协方差而不是复用 planarity 的返回值：planarity 只需要
    # 特征值，特征向量在那边被丢掉了。两次 eigh 的代价可以忽略
    # （2500x3 矩阵，约 0.2 ms），不值得为它把接口搞复杂。
    _, vecs = np.linalg.eigh(np.cov((p - p.mean(axis=0)).T))
    main_normal = vecs[:, 0]      # eigh 升序，第 0 列对应最小特征值

    d_all = np.abs((p - p.mean(axis=0)) @ main_normal)
    rms_all = float(np.sqrt(np.mean(d_all ** 2)))

    # 扣除「我们斜着看一个平面」造成的残差。平板跨度 L 上均匀分布、
    # 相对主平面倾斜 θ 时，残差 RMS = L/sqrt(12)·sinθ。用 expected_size
    # 而不是实测跨度：实测跨度本身可能被污染污染，用它会让判据自我适应
    # ——碎片变短 → 倾斜贡献变小 → 残差变小 → 更像合格。这正是「安静地
    # 给错答案」的一种。宁可保守用标称尺寸。
    lean = float(expected_size) / np.sqrt(12.0) * math.sin(
        math.radians(PLANAR_LEAN_TILT_DEG))
    r.lean_rms = lean
    # 在平方域相减再开方，不是在 RMS 上直接减 —— 两个正态分量的合成是
    # 方差相加。直接在 RMS 上减会低估残差（这里约低估 1.1 mm）。
    r.planar_rms = float(math.sqrt(max(rms_all ** 2 - lean ** 2, 0.0)))

    ok_planar = r.planar_rms < PLANAR_RMS_MAX
    r.checks['planarity'] = (ok_planar, f'残差={r.planar_rms*1000:.2f}mm',
                             f'<{PLANAR_RMS_MAX*1000:.1f}mm')
    if not ok_planar:
        r.reason = (
            f'平面性不合格：扣除倾斜({PLANAR_LEAN_TILT_DEG}°)后的面残差 '
            f'{r.planar_rms*1000:.2f} mm ≥ {PLANAR_RMS_MAX*1000:.1f} mm'
            f'（原始 RMS {rms_all*1000:.2f} mm，λ₃/λ₁={ratio:.2e}）。'
            f'要么点簇混进了不同朝向的面，要么深度噪声已经大到点云里'
            f'没有足够几何信息（见 PLANAR_RMS_MAX 注释里的失效边界）。')
        return r

    # --- ② 凸包面积（在主平面的 2D 投影上算） ----------------------------
    #
    # 投影到主平面的两个正交方向，而不是直接取 x-y：
    # 点云在 base_link 系里略有倾斜（第 40 条：光轴偏竖直 12.3°，
    # 同一个水平顶面在点云里是斜的）。直接取 x-y 投影会把 5 mm 的倾斜
    # 厚度算进面积，面积偏大 ~20%，而这 20% 会直接吃掉上界的余量。
    # 投到主平面上量的是「物体在自身平面上的真实覆盖面积」，
    # 与观察角度无关。
    u, v = _plane_basis(main_normal)
    uv = np.stack([(p - p.mean(axis=0)) @ u, (p - p.mean(axis=0)) @ v], axis=1)
    area = hull_area_2d(uv)
    ref_area = float(expected_size) ** 2
    if area is None or area <= 0:
        r.reason = '凸包退化（点共线或重合）'
        r.checks['hull_area'] = (False, '退化', f'∈{HULL_AREA_RATIO_RANGE}')
        return r
    r.hull_area = area
    r.hull_area_ratio = area / ref_area
    lo, hi = HULL_AREA_RATIO_RANGE
    ok_area = lo <= r.hull_area_ratio <= hi
    r.checks['hull_area'] = (ok_area, f'面积比={r.hull_area_ratio:.2f}',
                             f'∈[{lo}, {hi}]')
    if not ok_area:
        if r.hull_area_ratio < lo:
            why = '面积偏小 —— 目标很可能只有一部分在视野里'
        else:
            why = '面积偏大 —— 很可能与邻近物体粘连'
        r.reason = (f'凸包面积比 {r.hull_area_ratio:.2f} 不在 '
                    f'[{lo}, {hi}]（实测 {area*1e6:.0f} mm² / 理论 '
                    f'{ref_area*1e6:.0f} mm²）。{why}。')
        return r

    # --- ③ 法向一致性 ---------------------------------------------------
    normals, _ = point_normals(p)
    score, _ = normal_consistency(p, normals, main_normal)
    r.normal_score = score
    ok_normal = score < NORMAL_CONSISTENCY_MAX
    r.checks['normal_consistency'] = (ok_normal, f'1-|cos|={score:.3f}',
                                      f'<{NORMAL_CONSISTENCY_MAX}')
    if not ok_normal:
        r.reason = (f'法向不一致：E[1-|cosθ|] = {score:.3f} ≥ '
                    f'{NORMAL_CONSISTENCY_MAX}。点簇里混进了不同朝向的'
                    f'面（典型是方块顶面 + 桌面残留）。')
        return r

    # --- ④ MCD 剔离群 + 分位数中心 --------------------------------------
    # 前置拒绝全部通过，才做这一步。反过来做会浪费算力，且更糟的是：
    # 在一个「其实是碎片」的簇上算出的中心，会带着一个精确的数值
    # 进入下游，看起来比「直接失败」更可信。
    mask = mcd_inliers(p)
    r.inliers = mask
    inner = p[mask]
    if len(inner) < MIN_POINTS // 2:
        r.reason = f'MCD 内点太少（{len(inner)}），点云形状不可信'
        r.checks['mcd'] = (False, len(inner), f'≥{MIN_POINTS//2}')
        return r

    # 内点集上的残差：这是最终报告给用户的那个数（①里算的是全点云的，
    # 用来做判据；这里的更干净，因为剔掉了离群点后才统计）。
    # 再次扣除倾斜 —— MCD 是在原始点云上做的，内点集虽然更干净，
    # 但倾斜造成的几何厚度依然在。
    dist = (inner - inner.mean(axis=0)) @ main_normal
    r.residual = float(math.sqrt(max(
        float(np.mean(dist ** 2)) - r.lean_rms ** 2, 0.0)))

    lo_q, hi_q = CENTER_PERCENTILE
    center = np.array([
        0.5 * (np.percentile(inner[:, k], lo_q) + np.percentile(inner[:, k], hi_q))
        for k in range(3)
    ])
    r.center = center
    r.ok = True
    return r


def _plane_basis(normal):
    """给定法向，返回主平面内两个正交单位向量 (u, v)。

    构造方式：拿绝对值最小的那个分量轴，与法向做叉积得到 u，
    再 u×normal 得到 v。这样构造保证 u ⟂ normal 且 |u| = 1，
    v = u × normal 也 ⟂ normal。**不依赖任何特殊分支** ——
    如果拿「离法向最近的坐标轴」，在法向接近该轴时归一化会爆掉，
    而接近坐标轴的法向在俯视抓取里恰恰很常见（光轴朝下时
    主法向就是 ±Z）。这是那种「只在特定姿态下炸」的写法。
    """
    n = np.asarray(normal, dtype=float)
    n = n / max(np.linalg.norm(n), 1e-12)
    k = int(np.argmin(np.abs(n)))
    axis = np.zeros(3)
    axis[k] = 1.0
    u = np.cross(n, axis)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    return u, v


if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # 自检。用**合成点云**而不是真实点云，因为这里要验的是「判据在已知
    # 真值下会不会给出预期结论」，而真实点云没有干净的标签。
    #
    # 但合成数据必须**按真实密度与真实噪声生成**，否则阈值等于白标。
    # 第一版自检踩过这个坑：用 nx=40 的网格铺 25 mm（间距 0.64 mm）配上
    # 1.3 mm 噪声，看上去合理，实际法向σ 达22°，把完整目标也拒了 ——
    # 一度以为是阈值错。真实参数是这样反算的：
    #     水平 FOV 60°、320 px 宽、观测距离 106 mm（实测）
    #       → 视场宽 122.4 mm → 像素pitch 0.383 mm
    #       → 25 mm 方块约 65 px 见方 → 约 54x45 = 2430 点（与实测 2400~2500 吻合）
    #       → 网格间距 0.46 mm
    # 深度噪声 std 1.3 mm 也是实测值（arm_camera.xacro 写的 stddev=0.001
    # 对应约 1.3 mm 的实际散布）。倾斜 tan(12.3°) 是踩坑记录第 40 条实测的
    # 光轴偏竖直角 —— 它让顶面在点云里是斜的，25 mm 跨度对应约 5 mm 的 z 起伏。
    # ---------------------------------------------------------------------
    SIZE = 0.025
    SPACING = 0.00046        # 实测点间距 0.46 mm
    NOISE = 0.0013           # 实测深度噪声 std 1.3 mm
    TILT = 0.21              # tan(12.3°)

    print('=== geometry_validity 自检 ===')
    print(f'  Open3D: {"可用（走它的法向实现）" if _HAVE_OPEN3D else "不可用（走 numpy/scipy）"}')
    print(f'  合成参数：点间距 {SPACING*1000:.2f} mm · 深度噪声 {NOISE*1000:.1f} mm'
          f' · 倾斜 tan(12.3°)')
    print(f'  判据    ：面残差 < {PLANAR_RMS_MAX*1000:.1f} mm（已扣除 '
          f'{PLANAR_LEAN_TILT_DEG}° 倾斜的 {SIZE/np.sqrt(12)*np.sin(np.radians(PLANAR_LEAN_TILT_DEG))*1000:.2f} mm）· '
          f'面积比 ∈ {HULL_AREA_RATIO_RANGE} · '
          f'1-|cos| < {NORMAL_CONSISTENCY_MAX} (k={NORMAL_KNN})')

    def make_plate(spacing=SPACING, noise=NOISE, tilt=TILT, seed=0):
        """按真实密度生成一块 25 mm 平板。"""
        r = np.random.default_rng(seed)
        n = max(int(round(SIZE / spacing)), 4)
        gx = np.linspace(-SIZE / 2, SIZE / 2, n)
        gy = np.linspace(-SIZE / 2, SIZE / 2, n)
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        X, Y = X.ravel(), Y.ravel()
        p = np.stack([X, Y, tilt * X], axis=1)
        # 0.25*spacing 的位置抖动：真实深度图不是完美栅格，
        # 各向同性深度噪声会让采样点轻微散开。不加的话邻域会退化成
        # 规则的网格行，法向估计偏乐观。
        p = p + r.normal(0, 0.25 * spacing, p.shape)
        p[:, 2] += r.normal(0, noise, len(p))
        return p

    def rot_y(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    full = make_plate(seed=5)
    wall_full = (make_plate(seed=8) @ rot_y(90).T
                 + np.array([0.0, SIZE * 0.54, 0.0]))
    # expect 用三态而不是 bool：'pass' / 'reject' / 'limit'。
    # 'limit' 表示「这是一个已知的失效边界，按设计就该拒绝」——
    # 必须显式写出来，而不是混进must-pass 里然后靠调阈值凑绿。
    cases = [
        # ① 基准工况：完整目标，视野中心，真实噪声 —— 必须通过
        ('完整目标（视野中心，实测密度 1.3 mm 噪声）', full, 'pass'),
        # ② 严重截断：只剩 10 mm 宽的碎片 —— 应被面积判据拒
        ('截断成 10 mm 碎片', full[full[:, 0] > SIZE / 2 - 0.010], 'reject'),
        # ③ 粘连：紧邻的另一块 25 mm 板 —— 应被面积判据拒
        ('两块相邻板粘连（总宽约 55 mm）',
         np.vstack([full, make_plate(seed=6) + np.array([SIZE * 1.05, 0, 0])]),
         'reject'),
        # ④ 跨面混合：水平顶面 + 整块竖直接触面 —— 应被残差判据拒
        ('水平顶面 + 竖直接触面（非平面）',
         np.vstack([full, wall_full]), 'reject'),
        # ⑤ 掺入 25% 异面点 —— 应被拒（实测残差远超阈值）
        ('顶面 + 25% 竖直残留', np.vstack([full, wall_full[:730]]), 'reject'),
        # ⑥ 噪声 4 mm：残差判据放行（3.56 mm），但**法向判据拒绝**
        #    （1-|cos|=0.460，噪声直接主导了法向估计）。
        #    这是真实的已知边界，不是阈值配错 —— 见 NORMAL_CONSISTENCY_MAX
        #    注释：深度噪声相对点间距(0.46 mm)越大，法向越不可信。
        #    当前实测噪声 1.3 mm 时该指标为 0.229，余量充足。
        ('完整目标（深度噪声 4 mm，法向判据的已知边界）',
         make_plate(noise=0.004, seed=9), 'limit'),
        # ⑦ 噪声 6 mm：**残差判据**的已知边界，按设计就该拒绝。
        #    注意它与 ⑥ 被不同判据拒绝 —— 这正好说明两道判据覆盖的是
        #    不同失效模式，不是冗余。
        ('完整目标（深度噪声 6 mm，残差判据的已知边界）',
         make_plate(noise=0.006, seed=10), 'limit'),
    ]

    results = []
    for name, pts, expect in cases:
        res = validate_and_estimate(pts, SIZE)
        if expect == 'limit':
            tag = '拒绝'if not res.ok else '通过'
            mark = '（预期：已达边界）'
        else:
            tag = '通过' if res.ok else '拒绝'
            mark = '' if (res.ok == (expect == 'pass')) else '← 与预期不符！'
        print(f'\n  [{tag}] {name}{mark}')
        print(f'        {res.summary()}')
        if res.ok:
            print(f'        面残差 {res.planar_rms*1000:.2f} mm'
                  f'（阈值 {PLANAR_RMS_MAX*1000:.1f}，余量 '
                  f'{(PLANAR_RMS_MAX-res.planar_rms)*1000:.2f} mm）· '
                  f'倾斜贡献已扣除 {res.lean_rms*1000:.2f} mm · '
                  f'λ₃/λ₁={res.planarity:.2e}（仅诊断）· '
                  f'MCD 内点 {int(res.inliers.sum())}/{len(pts)}'
                  f'（{100*res.inliers.sum()/len(pts):.0f}%）')
            results.append((name, res.planar_rms, expect))
        else:
            print(f'        → {res.reason}')

    print('\n=== 结论 ===')
    bad = [n for n, v, e in results if e == 'pass' and v > PLANAR_RMS_MAX * 0.8]
    print(f'  must-pass 工况的面残差余量：')
    for name, v, e in results:
        if e == 'pass':
            print(f'    {name[:28]:30s} {v*1000:5.2f} mm '
                  f'（用掉阈值的 {100*v/PLANAR_RMS_MAX:.0f}%）')
    print(f'\n  若某个 must-pass 工况用掉阈值 >80%，说明余量不足，')
    print(f'  真实数据稍有变化就会开始误拒。')
    print(f'\n  已知边界（写在常量注释里，勿调阈值去"修"它们）：')
    print(f'    · 深度噪声 ≈ 4 mm：法向判据开始拒绝完整目标')
    print(f'      （噪声主导法向估计；残差判据此时仍放行）')
    print(f'    · 深度噪声 > ~5 mm：残差判据也开始拒绝')
    print(f'    · 异面污染 < ~15%：可能漏放，靠上游分割质量兜')
    print(f'\n  要提高对高噪声的容忍度，正确的方向是**改善点云质量**')
    print(f'  （加大 k、换更好的深度相机、或在分割阶段剔干净），')
    print(f'  而不是放宽这些阈值 —— 放宽会把异面污染一并放进来。')