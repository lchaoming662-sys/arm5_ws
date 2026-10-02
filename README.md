# arm5_ws · O5-2 五自由度桌面机械臂

ROS 2 Humble + **Gazebo Fortress** 的机械臂描述 / 控制 / 仿真 / 抓取演示工程。

结构参照 `akabot`（参考实现）与《11 从零手敲代码实战》（说明手册）重建，
原始素材是 WSL 导出的单包 `~/arm_description`（Gazebo Classic）。
`~/arm_description` 与 `~/ros2_arm` **未被改动**，可随时对照。

---

## 文档

| 文件 | 内容 |
|---|---|
| **[docs/使用说明书.md](docs/使用说明书.md)** | 环境要求、目录结构、接口清单、启动方式、常用操作、设计决策、物理限制、验收清单 |
| **[docs/踩坑记录.md](docs/踩坑记录.md)** | 全程踩到的 32 个问题：真实报错原文 + 根因 + 判据 + 修法 |

## 快速开始

```bash
source /opt/ros/humble/setup.bash
source ~/arm5_ws/install/setup.bash

ros2 launch arm_bringup bringup.launch.py               # 一条命令：仿真 + 控制器 + MoveIt + RViz
/usr/bin/python3 ~/arm5_ws/tools/pick_demo_moveit.py    # MoveIt 抓取演示（需上面在跑）
```

想分步来（调试时）：

```bash
ros2 launch arm_description description_launch.py       # ① RViz 看模型（带关节滑块）
ros2 launch arm_gazebo gz_launch.py                     # ② 仿真（Fortress + 控制器）
ros2 launch arm_moveit_config demo.launch.py            # ③ MoveIt（需 ② 在跑）
/usr/bin/python3 ~/arm5_ws/tools/pick_demo_moveit.py    # ④ MoveIt 抓取演示（需 ②③ 在跑）
```

仿真想跑快一点（本机软件渲染，物理步进是瓶颈）：

```bash
ros2 launch arm_bringup bringup.launch.py gui:=false      # 只跑 server，不开界面
ros2 launch arm_bringup bringup.launch.py camera:=false   # 不加载腕部相机
ros2 launch arm_gazebo gz_launch.py gui:=false            # 只要仿真时同理
```

编译：

```bash
cd ~/arm5_ws && colcon build --symlink-install && source install/setup.bash
```

> 注意：本机还装着旧的 `~/arm_ws`（Gazebo Classic 版 `arm_description`）。
> **不要和本工作区同时 source**——同名包会导致 `package://arm_description/meshes/...` 解析不确定。

## 包结构

```
src/arm_description/    纯模型（arm_core.xacro）+ 惯量宏 + 相机 + 显示 / 真机总装
src/arm_bringup/        控制器配置 + **总装入口**（bringup.launch.py + 门闩脚本）
src/arm_gazebo/         仿真环境（世界 / 桥 / gz_launch）
src/arm_moveit_config/  MoveIt 2（SRDF / IK / 规划器 / RViz）
tools/                  MoveIt 抓取演示、位形求解、几何工具、控制器误差采样器
docs/                   说明书与踩坑记录
```

模型命名对齐参考工程 akabot：用**语义名**而不是编号
（`top_plate` / `lower_arm` / `upper_arm` / `wrist` / `claw_base` / `right_claw` / `left_claw`），
惯量统一走 `inertial_macros.xacro`。详见说明书「三·补、命名约定」。

## 已验证的结果

| 环节 | 结果 |
|---|---|
| 一条命令启动 | `ros2 launch arm_bringup bringup.launch.py` → 15 s 后 `You can start planning now!`（无界面、无相机） |
| MoveIt 抓取闭环 | 方块（25 mm / 20 g）被夹持**提离地面 135.1 mm**，放回误差 Δx = +0.83 mm |
| 笛卡尔路径 | 下压 / 抬起 / 放回四段 **fraction 全部 = 1.000** |
| MoveIt IK | `/compute_ik` 返回 `error_code=1`（5 自由度靠 `position_only_ik: true`） |
| MoveIt 规划+执行 | `/move_action` 关节残差 **全部 < 0.01 rad** |
| 抓取流程耗时 | **50 s**（无界面、无相机，RTF 更高；带 GUI 约 63 s）|

验收都不看 ROS 层的自述值：抓取读 Gazebo 内部位姿（`ign model -m target_cube -p`），
MoveIt 执行读执行后的 `/joint_states` 实测值。

## 已知限制

- 俯抓姿态（两指竖直朝下、指尖不破地面）的抓手高度窗口只有 **16 mm**
  （z ∈ [0.023, 0.039]）——这条臂**只能抓地面上的东西**，放 3 cm 高的台面上就够不着。
  详见说明书第七节。
- `effort` / `velocity` 全是**让仿真跑得起来的工程取值**，不是舵机真实能力，
  不得用于评估负载或抓取力，也不得写进论文当本机参数。
- 本机 IK 只有 **KDL**。接新的规划组时，只要自由度不足 6，
  `kinematics.yaml` 就必须设 `position_only_ik: true`，否则 IK 永远失败。
- 仿真 RTF 只有 **0.39**（软件渲染），瓶颈是**物理步长**不是相机 ——
  `arm_world.sdf` 里 `max_step_size` 已从 1 ms 放宽到 3 ms 换来 2.8 倍提速。
  再往上加会让接触解算变糙，**会反过来影响夹爪抓取**，别乱调。
- **已知问题**：抓取演示收尾的「MoveIt 规划回零位」可能报 `CONTROL_FAILED(-4)`，
  而抓放本身是成功的。真因是那一刻 **`wrist_joint` 物理上一动不动**
  （实测 12 s 内位移 0.0000，一直压在 -1.5708 的限位上），误差线性累积越过
  0.12 rad 容差被掐断。**与速度无关**（0.20 / 0.40 两档都失败）。
  失败时**重启仿真**即可。完整证据与已排除的原因见踩坑记录第 31 条，
  排查工具是 `tools/jtc_trace.py`。
- `real_arm.urdf.xacro` 结构就绪但**真机硬件插件待接**（`arm_bringup` 只做到仿真）。
- 注意：**动手测量前先确认只有一个仿真在跑**（`ps` 数一下 + 看 `/clock` 是不是 ~150 Hz）。
  两个仿真并行时 `/joint_states` 会被搞乱，测量**不会报错，只会安静地给错答案**——
  本次为此走了很长的弯路，见踩坑记录第 32 条。
