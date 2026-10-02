# pick_ik 对比测试报告（test/pick-ik-compare 分支）

日期：2026-10-03
分支：`test/pick-ik-compare`（自 main `517779d` 切出，**未合入主线**）
结论先行：**不切主线，KDL 保留。** 数据与依据如下。

---

## 一、测试设计

- **被测对象**：KDL（`position_only_ik: true`，yaml 现行配置） vs pick_ik 1.1.2
  （apt 装 `ros-humble-pick-ik`，插件 `pick_ik/PickIkPlugin`，mode=global，
  rotation_scale=0.0，其余参数按 kinematics.yaml 备选段）
- **测试方式**：`tools/ik_compare_bench.py` 直调 `/compute_ik`，不碰 MTC 流水线，
  不起 Gazebo（move_group 单独跑，`use_sim_time:=false`，`/joint_states`
  由脚本自发自收 home 零位）。计时走墙上时钟，不受 RTF 干扰。
- **公平性控制**：20 个目标确定性生成（x=0.0267，y∈±0.16 五档 × z∈0.14~0.30
  四档）、同一把显式种子（home 全零随请求下发）、同一请求超时 0.5s。
- **前置自检（每次运行都过）**：服务在线；`ros2 param get` 核对
  `robot_description_kinematics.arm.kinematics_solver` 确实是本次要测的求解器
  （配置写了≠被加载）；warmup 确认 move_group 拿到机器人状态。
- **测量可信度**：两次运行之间目标逐个相同、种子相同、超时相同；
  pick_ik 参数切换后用 `ros2 param get` 复核过（mode/rotation_scale/cost_threshold）。

## 二、网格测试结果（20 目标）

| 配置 | 成功率 | 成功解全关节最小余量¹ | 求解时间中位 | 备注 |
|---|---|---|---|---|
| KDL（现行） | **15/20** | 0.0653 rad（3.7°） | 0.6 ms | 5 个失败全在 y<0 低 z 区 |
| pick_ik avoid=0（基线） | **15/20** | **0.0001 rad（贴死硬限位）** | 11.9 ms | 失败目标与 KDL 完全相同 |
| pick_ik avoid=1, ct=1e-4（yaml 原样） | 8/20 | 0.8211 rad | 12.2 ms | 只剩 y=±0.08/+0.16 列 |
| pick_ik avoid=1, ct=0.3 | 8/20 | 0.8348 rad | 11.4 ms | 同上 |
| pick_ik avoid=0.1, ct=0.03 | 8/20 | 0.8253 rad | 11.1 ms | 同上 |

¹ 余量 = 1.5708 − |q|（对 URDF 硬限位 ±1.5708）。

关键读数：

1. **两种求解器对"哪些目标不可达"完全一致**（pick_ik global 不依赖种子，
   仍解不出同样 5 个）——那 5 个目标是真不可达，不是 KDL 的锅。
2. **avoid=0 时 pick_ik 的解比 KDL 更贴限位**（0.0001 vs 0.065）：
   KDL 顶的是 joint_limits.yaml 的软限位 1.50（余量 0.0708），
   pick_ik 的 memetic 采样直接顶到 URDF 硬限位。光换求解器治不了顶限位。
3. **avoid>0 时成功目标集缩水到 8 个**：代价函数在适应度里制造
   "偏离目标、关节居中"的假极小——对只存在 margin<0.785 rad 构型的目标
   （y=0 一列），优化器收敛到折中点、位置检验过不了、反复重启到超时。
   权重从 1.0 降到 0.1 也救不回来（位置误差项在解附近 ~1e-6，
   任何非零避限位项都盖过它）。

## 三、真实抓取目标探针（`tools/ik_probe.py`）

| 目标 | KDL | pick_ik (avoid=0.1, ct=0.03) |
|---|---|---|
| y=+0.12, z=0.20（踩坑 17 条验证目标） | ✅ 3.6ms，余量 1.42/1.57 | ✅ 28.7ms，余量 1.47/0.98 |
| y=+0.12, z=0.25 | ✅ 2.3ms，余量 1.55/1.57 | ✅ 26.9ms，余量 1.20/1.31 |
| y=+0.05, z=0.20 | ✅ 1.6ms，余量 1.12/1.57 | ✅ 43.2ms，余量 0.79/1.57 |
| y=−0.05, z=0.20 | ✅ 1.5ms，wrist 余量 **0.0925** | ❌ −31 |

- 当前设计抓取目标带（y=+0.05~+0.12）两边都能解，pick_ik+avoid 的解
  离限位确实更远；但 y=−0.05（左手侧）pick_ik+avoid 直接无解，KDL 能解
  （且它那个解 wrist 余量 0.0925，正是离限位最近的解——代价函数想治的场景，
  pick_ik 反而给不出任何解）。
- KDL 时间 1.5~3.6ms，pick_ik 27~43ms：都不构成瓶颈，但 pick_ik 失败时
  烧满 0.2s 超时（20 核给 4 线程的 memetic 反复重启）。

## 四、上游源码结论（1.1.2 与 main 的 goal.cpp/robot.cpp 一致）

- `avoid_joint_limits` 代价形状：`Σ [ fmax(0, |p−mid|·2 − half_span) · f ]²`，
  其中每个关节的 `f = minimal_displacement_factor`（无速度定义时 = 1/N = 0.2，
  有速度时按最大速度归一）。**关节在区间中间半程（余量 > 0.785 rad）时代价恒为 0**。
- 接受条件是硬门：每个 goal 的 `cost × weight² < cost_threshold²`。
  权重与门控**共享同一个 weight**，无法解耦。
- 门控数值换算（f=0.2）：贴死 0.0987 / 设计位形（余量 0.105）0.0741 / 居中 0。
  ct=0.03（w=0.1）确实能分开"设计位形接受、贴死拒绝"——实测验证了。
  但这个窗口只有 1.33 倍，且与适应度耦合，实用上太脆。
- 自定义代价函数必须从 C++ 侧 `RobotState::setFromIK()` 传 `IkCostFn`；
  **`/compute_ik` 服务路径带不进任何自定义代价**。
  `tools/joint_limit_cost.py` 的 `register_limit_cost()` 注释也自认：
  Python 绑定没暴露这个接口，它从未真正注册过。
- **上游 PickNik 已于 2026-09-17 官方弃用 pick_ik**
  （"performance did not meet our requirements"，只做基础维护，寻找社区接管）。
  README 声称的"避免关节限位代价函数"实际存在但如上所述有结构性局限。

## 五、工程判断

**不切主线。**理由：

1. 收益不成立：KDL 在成功率（15/20 vs 8/20）、时间（0.6 vs 11ms）、
   真实抓取目标覆盖（4/4 vs 3/4）上全面不输；pick_ik+avoid 唯一的赢面
   （解离限位 0.82+ rad）只在它能解的子集上成立。
2. 代价真实存在：换求解器后所有轨迹变化，手眼标定、抓取姿态、闭合角 −0.15、
   峰值抬升 30mm 判据全要重验（几十分钟仿真）；而收益被第 2、3 条实测否定。
3. 维护风险：上游已弃用。
4. Task 4 的原始诉求（解别贴限位）已有工程正确性覆盖：
   joint_limits.yaml 软限位 ±1.50 让 KDL 顶不到硬限位
   （本次实测 KDL 最小余量 0.0653 rad）；若未来确实需要"连续偏好"，
   正确路径是 C++ 侧实现 `IkCostFn`（窗口形状按 joint_limit_cost.py 的
   SOFT_LIMIT_WINDOW 设计），且只挂在抓取节点的 setFromIK 调用上，
   不动全局 kinematics.yaml——那是另一个量级的工程，需要新的立项理由。

### 本分支保留的资产

- `tools/ik_compare_bench.py`：20 目标受控对比基准（含前置自检）
- `tools/ik_probe.py`：任意目标单点探针
- `docs/踩坑记录.md` 第 54 条：yaml 备选段漏配 avoid_joint_limits_weight +
  avoid 代价的假极小失效模式
- kinematics.yaml：仅多一行分支标记注释，配置实质未动（回退 = 删一行）

### 复现命令

```bash
# 1) 装 pick_ik（pkexec 弹窗认证）
# 2) KDL 基线
ros2 launch arm_moveit_config move_group.launch.py use_sim_time:=false
python3 tools/ik_compare_bench.py --label kdl --out /tmp/kdl.json \
  --expect-solver KDLKinematicsPlugin
# 3) 切 pick_ik：yaml 反注释 pick_ik 段 → colcon build → 重启 move_group
ros2 param set /move_group robot_description_kinematics.arm.avoid_joint_limits_weight 0.1
ros2 param set /move_group robot_description_kinematics.arm.cost_threshold 0.03
python3 tools/ik_compare_bench.py --label pickik --out /tmp/pik.json \
  --expect-solver PickIkPlugin
# 4) 真实抓取目标探针
python3 tools/ik_probe.py
```
