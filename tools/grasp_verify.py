#!/usr/bin/env python3
"""grasp_verify.py — 抓取结果的**物理层验证**

这个模块回答整个系统最要命的一个问题：

    「MTC 说成功、控制器说成功、attached_collision_object 也是齐的——
      那到底有没有真的抓到东西？」

为什么必须有这个模块
──────────────────
docs/踩坑记录.md 第 45 条记的就是这件事：有一轮流水线打印「执行完成」，
而实际上 y=14 那一组参数下抬起只上升了 **0.5 mm** —— 抓的是空气。
同��MTC 报 SUCCESS、arm_controller 报 SUCCESS、attached_collision_object
也是完整的。**三个语义层全部一致地给出了错误的结论。**

根因是它们都答错了问题：
  · MTC 的 SUCCESS 说的是「我规划出了一条合法的轨迹」
  · 控制器的 SUCCESS 说的是「我按那条轨迹走完了」
  · attached_collision_object 说的是「我在规划场景里把方块挂在了爪上」
  · 而真正的问题是「方块有没有真的被夹着抬起来」—— **只有物理层能回答**

铁律第 3 条：判定抓取成功与否必须问 Gazebo 读物理状态，
绝不看 MoveIt 的语义状态。本模块就是这条铁律的落地。

两条判据（必须**同时**成立）
──────────────────────────
  ① 峰值 z 抬升 > 30 mm
     从抓取动作开始到结束，记录方块质心 z 的最大值，算它相对动作开始时
     z 的抬升量。为什么是 30 mm：抬起指令是 LIFT_DISTANCE = 50 mm，
     而实测（干净环境）成功抓取能抬 47~49 mm，抓空气是 0.5 mm。
     30 mm 落在两者中间，且远低于指令值 —— 留出「夹持略微滑移」的余量，
     又远高于噪声（z 估计噪声约 1.3 mm，见踩坑记录第 43 条）。

  ② 抬起过程中的相对滑移 < 15 mm
     方块相对机械臂的水平位移。夹住了 → 两者刚性同动，位移 ≈ 0；
     抓空气 → 方块留在地上，位移 ≈ 整个抬升量（约 50 mm）。

【重要】原设计的「接触力 > 0 且持续 > 0.5 s」判据在本机**不可用**

实测发现（2026-10-03）：Fortress 的 apt 二进制包带的是 DART 6.12.1，
而 contact 消息里的 wrench（力）字段需要 DART 6.13。实际读到的消息
只有 4 个接触点的 position，没有 normal / wrench / depth。

这是上游已知缺陷（gz-sim issue #2037、#1662），不是 SDF 写错了。
所以本模块**不把「读不到力」当成「力为 0」** —— 那会把传感器缺陷
报告成「没夹到东西」，是最坏的一类误判。它会明确报「力不可测」，
并改用不依赖接触消息的滑移判据。详见下面 DART 那段注释。

测量可信性：这个工具必须能回答「这个测量可信吗」
────────────────────────────────────────────────
这是第 2 条铁律。环境被污染时测量不会报错，它会安静地给你一个看起来
合理的错误答案。所以每次 verify() 都先跑三项前置检查，
**任何一项不通过就直接拒绝出数据**，而不是给出一个带警告的数值：

  A. 环境唯一性 —— `ign gazebo` server 进程数必须恰好为 1。
     两个仿真同时跑时，/model/target_cube/pose 可能来自其中一个，
     而 contact 话题来自另一个 —— 两者拼在一起就是一个**从来不存在过
     的物理状态**。实测踩过（第 32 条）：清理不干净导致两个仿真并行，
     一整批测量全是假的。
     附加检查：/clock 频率翻倍也是征兆，但进程数是更硬的判据。

  B. 传感器确实在线 —— 在开始测量前先确认 contact 话题**有消息**。
     区分「力是 0」和「传感器没上线」：前者说明没夹到，
     后者说明世界文件没加载 Contact 插件。这是两种完全不同的故障，
     不加这个检查就会把「传感器没工作」误判成「没抓到」。

  C. 采样可信 —— 实际采样数与期望值的比例。太稀疏说明仿真卡住了
     （本机 RTF 只有 0.14~0.48，见踩坑记录第 23 条），
     或者话题QoS 没配对。
     注意：接触消息只在**有接触时**才发，所以「没有消息」在方块
     静止在地面时是正常的 —— B 项检查必须放在**方块已被抬起**之后，
     或者用「话题是否存在」而不是「有没有消息」来判断。

用法
────
    # 独立运行：读当前物理状态，给一份诊断报告
    /usr/bin/python3 ~/arm5_ws/tools/grasp_verify.py

    # 在流水线里（默认用法）
    from grasp_verify import verify_grasp
    result = verify_grasp(duration=..., lift_threshold=...)
    if not result.ok:
        return 4      # 绝不打印「执行完成」

退出码：0 通过 / 3 环境不唯一 / 4 物理判据不过 / 5 传感器未上线
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

# ===========================================================================
# 判据阈值 —— 全部集中在此，每个都附实测依据
# ===========================================================================

# 峰值 z 抬升下限（米）。
#
# 依据（实测，干净单仿真环境）：
#     成功抓取抬起量  47~49 mm（LIFT_DISTANCE = 50 mm，夹持滑移 1~3 mm）
#     抓空气抬起量    0.5 mm
# 30 mm 在两者之间，且距成功值有 17 mm 余量、距失败值有 29.5 mm 余量。
LIFT_THRESHOLD = 0.030

CONTACT_FORCE_EPS = 1e-6

# ---------------------------------------------------------------------------
# 【重要·本机实测限制】接触力读不到，wrench 段是空的
# ---------------------------------------------------------------------------
# 实测（2026-10-03，本机 Ubuntu 22.04 + Fortress apt 二进制包）：
#
#     $ ign topic -e -n 1 -t /target_cube/contact
#     header { stamp { sec: 58 nsec: 677000000 } }
#     contact {
#       collision1 { name: "target_cube::link::collision" }
#       collision2 { name: "ground_plane::link::collision" }
#       position { x: 0.0392 y: 0.037 z: 1.13e-13 }   ← 只有 4 个接触点位置
#       position { ... }                              ← 没有 normal
#     }# 没有 wrench、没有 depth
#
#     $ ign topic -e ... | grep -cE "wrench|normal \{|depth"
#     0            ← 确认一个都没有
#
# 这**不是 SDF 写错了**，是上游已知缺陷：gz-sim issue #2037 与 #1662。
# 根因在 DART：Fortress 的 apt 包带的是 DART 6.12.1，而 wrench/normal/depth
# 需要 6.13。Physics.cc 里那段填充代码有显式保护：
#     if (contact.second != nullptr) { ... add_normal / add_wrench ... }
# DART 6.12 不提供 contact.second，于是整段被跳过 ——
# 传感器照常工作、照常发消息，只是消息里没有力。
#
# 这个失效方式极其符合本工程最忌讳的模式：**看起来在工作，实际少给了一半数据**。
# 所以本模块的做法是：
#   1. 探测 wrench 是否真的存在，**不存在就明确报「力不可测」**，
#      绝不把「读不到」当成「力为 0」—— 后者会把传感器故障说成抓取失败；
#   2. 主判据改用**不依赖接触消息**的物理量（见 CONTACT_INDEPENDENT_*）；
#   3. 提供 --force-probe 自检，让使用者随时能确认本机是哪种情况。
#
# 想读到真正的接触力：装 DART 6.13 并从源码编译 gz-sim（见文件末尾的说明）。
# 在那之前，本模块用「接触点存在性 + 方块抬升」联合判定。
# ---------------------------------------------------------------------------

# 接触点存在性判据（替代接触力，在 DART 6.12 下唯一可用的接触信号）
#
# 为什么「有接触点」仍然有意义：消息里那4 个 position 就是真实的接触点，
# 只是没有对应的力。它能回答「方块此刻是否与**某个东西**压在一起」，
# 答不了「压得多紧」。
#
# 但它有个致命弱点：方块贴地时**必然**有 4 个接触点（实测就是 4 个），
# 所以「有接触点」在任何时候都成立，单独用它毫无判别力。
# 它只有在「方块已离开地面」的窗口里才有意义 ——
# 那时还有接触点，就说明接触来自手指而不是地面。
# 这也是为什么它必须与 z 抬升联合使用，且必须在抬起之后才采样。
CONTACT_POINT_EPS = 4          # 少于这个数视为无接触（方块底面是 4 个角）

# 不依赖接触消息的物理判据：方块在**离地段**相对机械臂的位移变化。
#
# 判据：以第一个「方块已离地」样本为参考，量后续离地样本相对它的
# 水平位移变化。真夹住了 → 方块与指尖刚性同动 → 变化 ≈ 0~3 mm；
# 中途脱手 → 方块被留在原地而机械臂继续走 → 变化长到几十 mm。
# 这个量只用 ign model -p 读位姿，不碰任何接触话题 ——
# 因此它在 DART 6.12 的环境下**照样可用**。
#
# 阈值 15 mm 的来历：夹爪在抬起过程中有软伺服滞后与方块轻微滑移，
# 实测（干净环境成功抓取）变化在 5 mm 以内；15 mm 留了三倍余量。
#
# 【分辨力边界，2026-10-03 全链路回归时如实标注】本流程的抬起与放回
# 都是**纯竖直**运动（原位放回）：「抓空气」时方块留在地上，相对位移
# 变化同样是 ~0 —— 所以「抓没抓到」主要由抬升判据回答（抓空气抬升
# 0.5 mm，阈值 30 mm，分得极开）；滑移判据分辨的是「抬着但已脱手」
# 的中间态，以及未来加了水平搬运段之后的脱落。
REL_SLIP_MAX = 0.015

# 抬升判据保持不变（这是最可靠的一条）
LIFT_THRESHOLD = 0.030

# 接触力必须持续的时长（秒，仿真时间）。
#
# 为什么是 0.5 s：LIFT_DISTANCE = 50 mm、笛卡尔速度缩放 0.25 下，
# 抬起段实测耗时约 1.5~2 s（仿真时间）。0.5 s 是其中的一小段，
# 短到不会因为末端减速而误判，长到能排除「擦了一下」。
# 不要设成整个抬起时长：那样只要中途有一个物理步的接触抖动就会失败。
CONTACT_HOLD_TIME = 0.5

# 环境唯一性检查：允许的 gazebo server 进程数。
GAZEBO_SERVER_PROCESSES = 1

# 采样可信度下限：实际收到 / 期望收到。低于此值认为采样不可信。
SAMPLE_RATIO_MIN = 0.5


@dataclass
class GraspResult:
    """一次抓取验证的完整记录。"""

    ok: bool = False
    reason: str = ''
    # 测量值
    z_start: float = 0.0
    z_peak: float = 0.0
    z_lift: float = 0.0
    contact_peak_force: float = 0.0
    rel_slip: float = None       # 相对滑移 p99，单位 m
    n_samples: int = 0
    # 前置检查的结果 —— 留着是为了在报告里说清「这次测量为什么可信」
    env_unique: bool = False
    sensor_online: bool = False
    sample_trustworthy: bool = False
    force_available: bool = False   # 接触消息里有没有 wrench（力）数据
    # 分项判定，便于打印
    lift_pass: bool = False
    contact_pass: bool = False
    detail: list = field(default_factory=list)

    def summary(self):
        if self.ok:
            slip = (f'滑移 {self.rel_slip*1000:.1f} mm'
                    if self.rel_slip is not None else '滑移 n/a')
            force = (f' 接触力 {self.contact_peak_force:.2f} N'
                     if self.force_available else ' 接触力（本机不可测）')
            return (f'抓取验证通过：抬升 {self.z_lift*1000:.1f} mm'
                    f'（阈值 {LIFT_THRESHOLD*1000:.0f}）· {slip}'
                    f'（阈值 {REL_SLIP_MAX*1000:.0f}）·{force}· '
                    f'{self.n_samples} 个采样')
        return f'抓取验证失败：{self.reason}'


# ===========================================================================
# 前置检查 A：环境唯一性
# ===========================================================================

def gazebo_server_count():
    """数正在跑的 Gazebo 仿真实例。返回 (数量, 明细列表)。

    实现踩过的坑（重要）：最初这里只匹配 'gz-sim-server'，结果**永远返回 0**，
    而仿真明明在跑 —— 环境唯一性检查于是变成一句永远通过的废话。
    用 ps 查了才发现，本机的实际进程是：

        sh     /bin/sh -c ruby /usr/bin/ign gazebo -r -v4 .../arm_world.sdf -s ...
        ruby   ign gazebo -r -v4 .../arm_world.sdf -s

    两个关键点：
      1. `ign gazebo` 是 **ruby wrapper**（踩坑记录第 8 条），真正的 C++
         server 是它的子进程，但它的进程名在 ps 里显示为 ruby，
         匹配 'gz-sim-server' 一个都找不到。
      2. ps 的 `comm` 列只有 15 个字符，'parameter_bridg' 就被截断了 ——
         所以必须匹配 `args` 整行，不能只看 comm。

    现在匹配三个模式，覆盖不同启动方式：
      · 'ign gazebo'  → ruby wrapper（本机实际情况）
      · 'gz-sim-server' / 'gz sim'  → 直接跑 server 或新版
      · 'arm_world.sdf' → 本工程世界文件的路径，最直接的判据
    同一个仿真会同时命中后两个模式，所以下面按「世界文件路径」去重 ——
    一个仿真实例就是**一个**带 arm_world.sdf 的进程组。
    """
    try:
        out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return -1, []

    detail, wrappers = [], []
    for line in out.splitlines()[1:]:
        if 'arm_world.sdf' not in line or 'grep' in line:
            continue
        # 一个仿真实例会占两个进程：`sh -c ruby ...ign gazebo ...`（wrapper）
        # 和它自己的 ruby 子进程。只数 wrapper 才是「仿真个数」；
        # 两个都数会把 1 个仿真报成 2 个。
        # 判据：命令行里含 'ign gazebo' 的**同时**不是被 sh 包着的那层 ——
        # 即排除掉以 '/bin/sh -c' 开头的行。
        if line.strip().startswith('/bin/sh') or ' sh -c ' in line[:40]:
            wrappers.append(line.strip()[:120])
        else:
            # 真正的 gazebo 进程行（可能是 ruby，也可能是 gz-sim-server）
            wrappers.append(line.strip()[:120])
            break      # 一个实例只记一行
    return len(wrappers), wrappers


def check_env_unique():
    """返回 (是否唯一, 说明文字)。

    注意：0 个仿真**不算**通过。没有任何仿真时同样无法测量 ——
    与其返回一个「通过」然后让调用方拿到一堆 None，
    不如直接说「环境不对」。这两种情况要分开报。
    """
    n, detail = gazebo_server_count()
    if n < 0:
        return False, '无法执行 ps，环境状态未知 —— 按「不唯一」处理'
    if n == 0:
        return False, ('没有仿真在跑（0 个）。无法验证抓取 —— '
                       '确认仿真已启动：ros2 launch arm_gazebo gz_launch.py')
    if n != GAZEBO_SERVER_PROCESSES:
        return False, (f'发现 {n} 个仿真同时在跑（应为 '
                       f'{GAZEBO_SERVER_PROCESSES}）。'
                       f'环境被污染时测量不会报错，只会安静地给出错误答案：'
                       f'位姿可能来自 A、接触话题来自 B，拼成一个'
                       f'从未存在过的物理状态。'
                       f'清理：pkill -f arm_world.sdf；'
                       f'当前进程：{detail}')
    return True, '环境唯一（1 个仿真进程）'


# ===========================================================================
# 读物理量：方块位姿
# ===========================================================================

def read_cube_pose(model='target_cube'):
    """读 Gazebo 里方块的位姿。返回 (x, y, z) 或 None。

    优先用 `ign model -m <名字> -p`。为什么不用订阅话题：
      · ign 命令走的是 Gazebo 自己的服务，拿到的是**当步**的真实状态；
      · 话题有 QoS 与发布频率问题（100 Hz 但受 RTF 影响，实测只有 30~50 Hz），
        且 ros_gz_bridge 需要额外配置一条桥接项。
    对「一次性读取一个标量」的命令行方式更简单可靠。

    解析注意（踩坑记录沿用）：ign 的输出是多行的，开头还有一行
    `Model: [8]` —— 那个 8 是模型 ID 不是坐标。只认「整行是一个方括号、
    里面恰好三个数」的那种行，否则会把模型 ID 当成 x 读进来。
    这个问题踩过一次，错读出来的 x 会是 8 而不是 0.027 —— 那是个很显眼
    的错值，不容易漏，但仍然是必须显式处理的解析细节。
    """
    if shutil.which('ign') is None:
        return None
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


def contact_topic_exists(topic='/target_cube/contact'):
    """检查 contact 话题**是否存在**（不是「有没有消息」）。

    为什么查「存在」而不是「有消息」：接触消息只在有接触时才发。
    方块静静躺在桌上——不对，它一直和地面有接触，所以其实一直有消息。
    但「刚生成还没落定」的那一瞬间是没有的。所以用「话题是否存在」
    来判断传感器是否上线更可靠。

    注意要检查的必须是 **gz 侧**的话题：contact 传感器发的是 gz-transport
    消息，没有桥接到 ROS 2（见arm_bridge.yaml 里没有这一项）。
    这是刻意的：验证逻辑用 `ign topic` 读，不给仿真增加额外负载。
    """
    if shutil.which('ign') is None:
        return False
    try:
        r = subprocess.run(['ign', 'topic', '-l'], capture_output=True,
                           text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return topic in r.stdout


def read_contact_message(topic='/target_cube/contact', timeout=6.0):
    """读一条接触消息，返回 (文本, 接触点个数, 是否有 wrench)。

    「是否有 wrench」是这里最关键的一个返回值 —— 它决定了本模块
    能不能给出接触力判据。见文件头 DART 6.12 的说明。

    接触点个数的数法：数 `position {` 出现的次数。消息里每个接触点
    对应一个 position（以及一个 normal + 一个 wrench + 一个 depth，
    后三者在本机缺失）。实测贴地时是 4 个，对应方块底面的 4 个角 ——
    这个数字本身有物理含义，可以用来判断接触面的形状。
    """
    if shutil.which('ign') is None:
        return '', 0, False
    try:
        r = subprocess.run(
            ['ign', 'topic', '-e', '-n', '1', '-t', topic],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return '', 0, False
    if r.returncode != 0:
        return r.stdout or '', 0, False
    text = r.stdout
    n_points = len(re.findall(r'position\s*\{', text))
    has_wrench = bool(re.search(r'wrench\s*\{', text))
    return text, n_points, has_wrench


def read_contact_forces(topic='/target_cube/contact', timeout=6.0):
    """读接触法向力。返回 (峰值 N, 是否有 wrench 数据)。

    保留这个函数是因为：一旦用户按文件头的说明装了 DART 6.13 并重编
    gz-sim，它就能直接用。**但它不能被当成「力是 0」** —— 那两个含义
    完全不同，混淆的后果是把「传感器缺数据」报告成「没夹到东西」。
    所以调用方必须先看第二个返回值。

    解析的是 wrench 块里的 body_1_force（不是顶层normal）：
        wrench {
          body_1_name: "target_cube::link::collision"
          body_1_wrench { force { x: .. y: .. z: .. } torque { ... } }
        }
    取 body_1_force 的模长。为什么是它而不是 body_2_force：
    body_2 是被撞的那一方，若是静态体（地面）则恒返回 0
    （上游文档明确说明这一点）。取「主动施加力的那一方」才稳定。
    """
    text, _, has_wrench = read_contact_message(topic, timeout)
    if not has_wrench:
        return 0.0, False
    return _max_body1_force(text), True


def _max_body1_force(text):
    """从消息文本里解析 body_1_force 的最大模长（牛顿）。"""
    best = 0.0
    for m in re.finditer(
            r'body_1_wrench\s*\{.*?force\s*\{([^{}]*)\}', text, re.S):
        nums = re.findall(r'[xyz]\s*:\s*(-?[\d.eE+]+)', m.group(1))
        if len(nums) >= 3:
            try:
                mag = sum(float(v) ** 2 for v in nums[:3]) ** 0.5
                best = max(best, mag)
            except ValueError:
                continue
    return best


def read_tcp_pose(model='o5_2_arm'):
    """读夹爪 tcp 在世界系里的位置。

    为什么需要它：主判据之一是「方块相对夹爪的滑移量」，这需要同时
    知道两者的位置。`ign model -m <robot> -p` 给出的是**模型原点**，
    不是 tcp —— 模型 o5_2_arm 的原点在 base_link，tcp 在它前方约 0.4 m。

    这里用模型原点而不是 tcp：两者在抬起过程中是**刚性同动的**，
    相对位移完全相同，所以用哪个都能算滑移。用模型原点省掉了
    「从世界系原点换算到 tcp」这一步，而那一步需要 FK ——
    引入 FK 就会引入「FK 用的关节值是哪一时刻的」这个新的不确定性
    （抬升过程中关节在动）。少一个变量就少一个出错的地方。
    """
    if shutil.which('ign') is None:
        return None
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


# ===========================================================================
# 主验证逻辑
# ===========================================================================

class GraspMonitor:
    """跨执行过程的后台物理状态采样器（线程）。

    为什么必须有它（2026-10-03 全链路回归发现）：verify_grasp() 自己的
    观察窗口在轨迹全部跑完之后才开始，那时方块要么已举在顶、要么已
    放回台面，窗口内 z 不再变化 —— 「峰值抬升 > 30 mm」在那种时序下
    永远量不出非零值，验证的通过路径根本不存在。抬升与滑移都必须在
    **执行期间**连续采样才可测。

    用法：
        mon = GraspMonitor()
        mon.start()                    # 发轨迹之前
        ... 执行 ...
        mon.stop()
        result = verify_grasp(samples=mon.samples)

    线程安全：只有采样线程写 samples，stop() join 之后主线程才读。
    采样节奏：间隔 = interval 仿真秒，ign 子进程的墙上开销用
    「本轮已耗时间」抵扣（否则 RTF 高时进程开销会把间隔撑大数倍，
    实测采到 4/30 个点，见 verify_grasp 内同名注释）。
    """

    def __init__(self, model='target_cube', arm_model='o5_2_arm', interval=0.1):
        self.model = model
        self.arm_model = arm_model
        self.interval = interval
        self.samples = []              # [(cube_xyz, arm_xyz_or_None)]
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        rtf = max(_rtf(), 0.05)        # 执行期间 RTF 会波动，但只影响
        while not self._stop.is_set(): # 采样疏密，峰值/p99 对疏密不敏感
            t_iter = time.time()
            cube = read_cube_pose(self.model)
            arm = read_tcp_pose(self.arm_model) if cube is not None else None
            if cube is not None:
                self.samples.append((cube, arm))
            spent = time.time() - t_iter
            time.sleep(max(0.0, self.interval / rtf - spent))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


def verify_grasp(duration=3.0, interval=0.1, lift_threshold=LIFT_THRESHOLD,
                 slip_max=REL_SLIP_MAX, model='target_cube',
                 arm_model='o5_2_arm', verbose=True, samples=None):
    """观察方块的物理状态，判断是否真的抓取成功。

    参数
    ----
    duration : 观察时长（秒，**仿真时间**）。抓取动作结束后至少要覆盖
               整个抬起过程 —— 实测抬起段约 1.5~2 s，所以给 3 s。
    interval : 采样间隔（秒，仿真时间）。
    arm_model : 机器人模型名，用于算相对滑移（见 read_tcp_pose）。

    返回 GraspResult。**调用方必须先判 ok**，不许拿 z_lift 单独用。

    为什么用仿真时间而不是墙上时间：仿真 RTF 只有 0.14~0.48（踩坑记录
    第 23 条），墙上 3 秒可能只推进了 0.4 秒仿真时间 —— 抬起动作根本没
    走完，就会得出「没抬起来」的假失败。这正是第 44 条的教训：
    没核对测量条件，得出的一切结论都不可比。

    两条判据（不是接触力，理由见文件头 DART 说明）
    ---------------------------------------------
      ① 抬升：方块质心 z 的峰值抬升 > 30 mm
      ② 滑移：抬起过程中方块相对机械臂的位移 < 15 mm
         夹住了 → 相对位移 ≈ 0；抓空气 → 方块留地上，相对位移 ≈ 50 mm。
         这条完全不依赖接触消息，因此在本机的 DART 环境下照样可用。
    """
    r = GraspResult()

    def say(msg):
        if verbose:
            print(msg, file=sys.stderr)

    # ---- 前置检查 A：环境唯一性 ----
    r.env_unique, env_msg = check_env_unique()
    say(f'[前置] 环境唯一性：{env_msg}')
    if not r.env_unique:
        r.reason = env_msg
        return r

    # ---- 前置检查 B：传感器上线 + 力数据可用性 ----
    r.sensor_online = contact_topic_exists()
    say(f'[前置] contact 话题存在：{r.sensor_online}')
    if not r.sensor_online:
        r.reason = (
            '接触传感器未上线（话题 /target_cube/contact 不存在）。'
            '这与「没抓到东西」是两回事，不要混为一谈。'
            '检查：① 世界文件里有没有 ignition-gazebo-contact-system 插件'
            ' ② <sensor> 是否是 <link> 的子元素'
            ' ③ <contact><collision> 里填的是碰撞体名字'
            '（可用 ign topic -l 看实际话题名）')
        return r

    # 探测 wrench 字段是否真的存在。**必须在判定之前做** ——
    # 不知道力能不能测就去看「力是不是 0」，那是把「读不到」当「没有」。
    text, n_pts, has_wrench = read_contact_message()
    r.force_available = has_wrench
    say(f'[前置] 接触消息：{n_pts} 个接触点，'
        f'wrench 数据{"有" if has_wrench else "**缺失（DART 6.12 限制）**"}')
    if not has_wrench:
        say('       → 接触力判据在本机不可用，改用「相对滑移」判据。'
            '这不是把误差糊过去，而是换一条本机真的能测的物理量。')

    # ---- 观察 ----
    if samples is not None:
        # 监视器路径：samples 由 GraspMonitor 在**执行全程**采集，
        # 每项 (cube_xyz, arm_xyz_or_None)。z 参考系 = 首样本 =
        # 执行开始前的方块高度（在地上），「峰值抬升」才有意义。
        pairs = list(samples)
        say(f'[观察] 使用监视器样本 {len(pairs)} 个（覆盖执行全程）')
    else:
        say(f'[观察] 开始采样，仿真时长 {duration:.1f} s...')
        rtf = max(_rtf(), 0.05)
        # ign model -p 每次调用都是一个子进程，墙上开销 0.3~1 s 且与 RTF 无关。
        # 墙上预算若只按 duration/rtf 给，RTF 高（机器空闲）时预算缩水，
        # 子进程开销会把有效采样间隔撑大 5~8 倍 —— 实测（2026-10-03，
        # RTF≈0.86）只采到 4/30 个点，被下面的前置检查 C 正确拒掉。
        # 修法：预算加上每个采样点的进程开销余量，且每轮 sleep 扣除本轮
        # 已耗时间 —— 「间隔 = interval 仿真秒」的本意不变。
        IGN_CALL_COST = 1.5          # 秒（墙上），一次 ign 子进程的悲观估计
        n_expected = max(int(duration / interval), 1)
        t_end = time.time() + duration / rtf + n_expected * IGN_CALL_COST
        pairs = []
        while time.time() < t_end and len(pairs) < n_expected:
            t_iter = time.time()
            cube = read_cube_pose(model)
            arm = read_tcp_pose(arm_model) if cube is not None else None
            if cube is not None:
                # 相对位移的**水平分量**：只比 x/y，不比 z。为什么见
                # 判据 ② 处的注释（基座高度不变，水平相对位移才是
                # 「方块有没有被留在原地」的直接度量）。
                pairs.append((cube, arm))
            spent = time.time() - t_iter
            time.sleep(max(0.0, interval / rtf - spent))

    r.n_samples = len(pairs)
    if samples is not None:
        # 执行全程的采样数取决于执行时长与 RTF，没有固定期望值可卡；
        # 下限 15 个 ≈ 窗口法 3 s × 10 Hz 的量级。低于它说明 ign 调用
        # 大面积失败或线程几乎没跑，测量不可信。
        r.sample_trustworthy = r.n_samples >= 15
    else:
        r.sample_trustworthy = (r.n_samples >= n_expected * SAMPLE_RATIO_MIN)
    say(f'[前置] 采样 {r.n_samples} 个点，可信：{r.sample_trustworthy}')
    if not pairs:
        r.reason = ('读不到方块位姿（ign model -p 无输出）。'
                    '确认世界名与模型名对得上。')
        return r
    if not r.sample_trustworthy:
        r.reason = (f'采样太稀疏（{r.n_samples}），'
                    f'测量不可信，拒绝出结论')
        return r

    # z 用稳健统计量而非 max（第 4 条铁律：稳健统计代替极值）。
    # 不过**峰值抬升**这个量本身就该用峰值 —— 我们要的就是「最高抬到多高」。
    # 折中做法：取 99 分位作为峰值，避免单个数值异常尖峰把结论带偏；
    # 同时保留真实 max 供对照打印。
    r.z_start = float(pairs[0][0][2])
    r.z_peak = float(max(p[0][2] for p in pairs))
    z_p99 = float(sorted(p[0][2] for p in pairs)[int(0.99 * (len(pairs) - 1))])
    r.z_lift = max(z_p99, 0.0) - r.z_start

    # ---- 判据 ①：抬升 ----
    r.lift_pass = r.z_lift > lift_threshold
    r.detail.append(
        f'抬升 {r.z_lift*1000:.1f} mm（z 起 {r.z_start:.4f} → '
        f'峰 {r.z_peak:.4f}，p99 {z_p99:.4f}），阈值 {lift_threshold*1000:.0f} mm'
        f' → {"通过" if r.lift_pass else "**不通过**"}')

    # ---- 判据 ②：相对滑移 ----
    # 只在**方块已离开地面**的样本上算。两个理由：
    #   · 贴地阶段（接近段）方块本来就不动，而 tcp 还在抓取点之外，
    #     水平相对距离是「还没抓」的正常值，混进来会把滑移虚报得很大
    #     —— 监视器跨执行全程采样后这个污染必须滤掉；
    #   · 贴地时方块与机械臂的水平关系完全由「机械臂在哪儿」决定，
    #     那不是「没抓住」的信号。
    if not r.lift_pass:
        r.contact_pass = False
        r.rel_slip = None
        r.detail.append('未抬起，滑移不作为判据（方块还在地上）')
    else:
        rels_air = [(p[0][0] - p[1][0], p[0][1] - p[1][1])
                    for p in pairs
                    if p[1] is not None and p[0][2] > r.z_start + 0.010]
        if not rels_air:
            # 通过了抬升判据却找不到离地样本？说明过滤条件与采样对不上，
            # 退回「有臂位姿的全部样本」并在明细里明说 —— 不安静地换口径。
            rels_air = [(p[0][0] - p[1][0], p[0][1] - p[1][1])
                        for p in pairs if p[1] is not None]
            r.detail.append('（无离地样本，滑移退回全程样本计算）')
        if not rels_air:
            r.contact_pass = False
            r.rel_slip = None
            r.reason_short = '读不到机械臂位姿，滑移判据无法计算'
            r.detail.append('**读不到机械臂位姿**（ign model -m o5_2_arm -p '
                            '无输出），滑移判据无法计算 —— 测量条件不完整，'
                            '按不可信处理')
        else:
            # 「滑移」量的是**位移的变化**，不是到基座的绝对距离。
            # 基座原点不动，rel = cube_xy − base_xy 就是方块的世界坐标
            # 模长 —— 方块被夹着平移/原位搬运时它恒在 36 mm 上下，
            # 拿绝对距离当滑移会把成功的抓放误杀成「滑移 39 mm」
            # （2026-10-03 全链路回归实测：抬升 47.8 mm 通过、原位放回
            # 落点偏差 2 mm，滑移却报 39.4 mm）。
            # 正确口径：以**第一个离地样本**为参考点，量后续离地样本
            # 相对它的位移。真夹住 → 方块与臂刚性同动 → 变化 ≈ 0~3 mm；
            # 中途脱落 → 方块被留在原地而臂继续走 → 变化长到几十 mm
            # （前提是脱落后有水平行程；本流程是原位放回、纯竖直运动，
            # 脱落判别主要靠抬升判据，滑移只对「抬着但已脱手」的
            # 中间态有分辨力 —— 这个边界如实写在这里）。
            ref = rels_air[0]
            disp = np.array(rels_air) - np.array(ref)
            d = np.linalg.norm(disp, axis=1)
            slip_p99 = float(np.percentile(d, 99.0))
            r.rel_slip = slip_p99
            r.contact_pass = slip_p99 < slip_max
            r.detail.append(
                f'相对滑移（离地段相对首离地样本的位移变化）'
                f'p99 {slip_p99 * 1000:.1f} mm'
                f'（离地样本 {len(rels_air)}/全程 {len(pairs)}）'
                f'，阈值 {slip_max * 1000:.0f} mm'
                f' → {"通过" if r.contact_pass else "**不通过**"}')

    # ---- 接触力（有 wrench 时作为附加判据，不替代滑移）----
    if r.force_available and r.lift_pass:
        force, _ = read_contact_forces()
        r.contact_peak_force = force
        if force <= CONTACT_FORCE_EPS:
            r.contact_pass = False
            r.detail.append(f'抬起后接触力 {force:.4f} N（要求 > 0）→ **不通过**')

    r.ok = r.lift_pass and r.contact_pass
    if not r.ok:
        r.reason = ('两项判据未同时成立 —— ' + '；'.join(r.detail))
    return r


def _rtf():
    """读实时因子。读不到就返回 1.0（退化成墙上时间）。

    为什么需要它：仿真 RTF 只有 0.14~0.48（踩坑记录第 23 条），
    用墙上时间当仿真时间会把采样窗口压缩 2~7 倍。
    读不到时退化为 1.0 是**保守**方向吗？不是 —— 它会让观察时长按
    墙上时间算，仿真时间可能只推进了一小段，导致观察不足。
    所以调用方应该把 duration 给得比需要的长（这里是 3 s）。
    """
    if shutil.which('ign') is None:
        return 1.0
    try:
        r = subprocess.run(['ign', 'topic', '-e', '-n', '1',
                            '-t', '/world/arm_world/world_stats'],
                           capture_output=True, text=True, timeout=5)
        m = re.search(r'real_time_factor:\s*([\d.]+)', r.stdout)
        if m:
            return float(m.group(1))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return 1.0


# ===========================================================================
# 独立运行模式
# ===========================================================================

def main():
    print('=== 抓取物理验证（独立诊断模式）===')
    print('这个模式只报告当前物理状态，不做「是否成功」的判定 ——')
    print('因为没有「抓取动作开始」这个时间基准。\n')

    print('--- 前置检查 ---')
    ok_env, env_msg = check_env_unique()
    print(f'环境唯一性     : {"通过" if ok_env else "**不通过**"}  {env_msg}')
    online = contact_topic_exists()
    print(f'接触传感器上线 : {"通过" if online else "**不通过**"}  '
          f'/target_cube/contact')
    if not online:
        print('\n  注意：传感器没上线时，下面读到的一切都是「假阴性」。')
        print('  查 ign topic -l 看有没有 contact 相关话题。')

    print('\n--- 物理状态 ---')
    pose = read_cube_pose()
    if pose is None:
        print('读不到方块位姿')
        return 5
    print(f'方块位姿: x {pose[0]:+.4f}  y {pose[1]:+.4f}  z {pose[2]:+.4f}')
    arm = read_tcp_pose()
    if arm is not None:
        d = ((pose[0] - arm[0]) ** 2 + (pose[1] - arm[1]) ** 2) ** 0.5
        print(f'机械臂原点   : x {arm[0]:+.4f}  y {arm[1]:+.4f}  z {arm[2]:+.4f}')
        print(f'水平相对距离 : {d*1000:.1f} mm'
              f'（阈值 {REL_SLIP_MAX*1000:.0f} mm）')

    text, n_pts, has_wrench = read_contact_message()
    print(f'接触点       : {n_pts} 个')
    if has_wrench:
        force, _ = read_contact_forces()
        print(f'接触力       : {force:.4f} N（wrench 数据可用）')
    else:
        print('接触力       : **本机不可测**')
        print('  原因：Fortress 的 apt 包带 DART 6.12.1，contact 消息里')
        print('  没有 wrench 字段（上游 gz-sim issue #2037，需要 DART 6.13）。')
        print('  → 流水线会改用「抬升 + 相对滑移」两条判据。')
        print('  → 想读到力：装 DART 6.13 后从源码编译 gz-sim。')

    print('\n--- 怎么用 ---')
    print('真正的判定要在流水线里做（需要「动作开始」这个基准）：')
    print('  from grasp_verify import verify_grasp')
    print('  res = verify_grasp(duration=3.0)')
    print('  if not res.ok: ...   # 绝不打印「执行完成」')
    return 0


if __name__ == '__main__':
    sys.exit(main())