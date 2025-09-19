# `G1 HugWBC` 扩展与奖励迁移提案

## 1. 训练框架现状与更新方案
### 1.1 当前 `g1` 训练框架的局限
- 环境 `references/envs/g1/g1_env.py` 仍继承 `LeggedRobot`，观测与奖励维度沿用早期平地行走设定，缺少跳跃/单脚跳所需的接触时序与高度信息。
- 配置 `G1RoughCfgPPO` 使用尺寸为 `[32]` 的单层 `MLP` 与 `ActorCriticRecurrent`，缺乏 `HugWBC` 在 `H1` 中的历史观测堆叠、潜在状态预测（`MlpAdaptModel`）与对称性约束，导致在快速离地任务中策略收敛慢、稳定性差。
- `rsl_rl/algorithms/ppo.py` 中的左右对称映射、`wbc` 对称损失系数均针对 `19 DoF` 的 `H1`，未为 `12 DoF` 的 `g1` 提供合适配置。

### 1.2 建议的更新步骤
1. **环境升级到 `HugWBC` 架构**：以 `legged_gym/envs/h1/h1.py` 为模板重构 `g1` 环境，改为继承 `BaseTask`，引入：
   - 历史观测缓冲（`ObservationBuffer`）与 `partial_obs_buf`，以满足 `MlpAdaptModel` 对时序输入的需求；
   - 接触期望 `self.desired_contact_states`、足端高度扫描与 `self.measured_foot_scan`，为脚离地奖励与跳跃控制提供支撑；
   - `standing_envs_mask`、`velocity_level` 等状态变量，支持站立/跳跃模式切换与速度约束奖励。
2. **指令与观测接口扩充**：扩展 `g1` 的 `commands` 维度，兼容 `HugWBC` 使用的 9 维命令（线速度 3 + 步态频率 + 步态相位 + 摆动高度 + 身体高度 + 身体俯仰 + 腰部滚转 + 方向），同时在观测中拆分 `PROPRIOCEPTION_DIM` / `CMD_DIM` / `PRIVILEGED_DIM`，保证与 `MlpAdaptModel` 一致。
3. **引入 `HugWBC` 策略模型**：在新增的 `G1HugWBCCfgPPO` 中：
   - 将 `policy.model_name` 设为 `"MlpAdaptModel"`，保留 `latent_dim`、`mlp_hidden_dims`、`sync_update` 等核心参数；
   - 依据 `g1` 观测维度调整 `critic_obs_dim` 与 `actor_hidden_dims`，推荐保持与 `H1` 相同的三层 `Actor`（[256,128,32]）与 `Critic`（[512,256,128]）。
4. **配置 `wbc` 对称损失**：为 `12 DoF` 的双足建立独立的关节/观测置换表，并在 `rsl_rl/algorithms/ppo.py` 中改为从任务配置读取置换矩阵，避免硬编码 `H1` 索引。
5. **训练流程调整**：
   - 使用 `HugWBC` 相同的 `OnPolicyRunner`，`num_steps_per_env` 维持在 24–32；
   - 将 `max_iterations` 提高到 20k 以上，以覆盖跳跃课题的收敛周期；
   - 在 `TaskRegistry` 中注册新任务名（如 `g1_hugwbc`），便于与旧版平地行走训练并行测试。

### 1.3 更新理由
- 历史观测 + 适应模型可显式记忆最近的接触与足端状态，为重复跳跃/交替单脚落地提供必要的时间上下文。
- 对称损失能在左右腿之间传递经验，减少单脚跳时的偏斜与围绕纵轴的漂移。
- 扩展命令向量让策略能接收显式的跳跃高度、摆动高度与步态模式指示，从而复用 `HugWBC` 的计划器与奖励设计。
- 统一的 `PPO` 配置（学习率自适应、`sync_update`）能稳定分布外行为（如空中停留），减少梯度爆炸与奖励震荡。

## 2. `HugWBC` 奖励函数继承结构分析
### 2.1 `LeggedRobot` 基类（`references/envs/base/legged_robot.py`）
| 奖励函数 | 作用简介 | 依赖接口/状态 | `g1` 迁移性 |
| --- | --- | --- | --- |
| `_reward_lin_vel_z` | 惩罚竖直速度 | `self.base_lin_vel[:,2]` | 直接复用 |
| `_reward_ang_vel_xy` | 惩罚横摆角速度 | `self.base_ang_vel[:,:2]` | 直接复用 |
| `_reward_orientation` | 保持躯干水平 | `self.projected_gravity` | 复用（需开启此尺度） |
| `_reward_base_height` | 约束腰部高度 | `self.root_states[:,2]`, `base_height_target` | 复用 |
| `_reward_torques` | 惩罚力矩大小 | `self.torques` | 复用 |
| `_reward_dof_vel` | 惩罚关节速度 | `self.dof_vel` | 复用 |
| `_reward_dof_acc` | 惩罚加速度 | `self.last_dof_vel`, `self.dt` | 复用 |
| `_reward_action_rate` | 控制动作变化 | `self.actions`, `self.last_actions` | 复用 |
| `_reward_collision` | 惩罚非足端碰撞 | `self.contact_forces`, `penalised_contact_indices` | 复用 |
| `_reward_termination` | 终止惩罚 | `self.reset_buf`, `self.time_out_buf` | 复用 |
| `_reward_dof_pos_limits` | 限制关节角 | `self.dof_pos`, `self.dof_pos_limits` | 复用 |
| `_reward_dof_vel_limits` | 限制关节角速度 | `self.dof_vel`, `self.dof_vel_limits` | 复用 |
| `_reward_torque_limits` | 限制接近力矩上限 | `self.torque_limits`, `self.torques` | 复用 |
| `_reward_tracking_lin_vel` | 跟踪线速度命令 | `self.commands[:,:2]`, `self.base_lin_vel` | 复用（需扩展命令维度） |
| `_reward_tracking_ang_vel` | 跟踪角速度命令 | `self.commands[:,2]`, `self.base_ang_vel[:,2]` | 复用 |
| `_reward_feet_air_time` | 奖励足端空中时间 | `self.contact_forces`, `self.last_contacts`, `self.commands` | 复用（跳跃关键） |
| `_reward_stumble` | 惩罚滑碰 | `self.contact_forces` | 复用 |
| `_reward_stand_still` | 静止时保持姿态 | `self.dof_pos`, `self.default_dof_pos`, `self.commands` | 复用 |
| `_reward_feet_contact_forces` | 惩罚足端冲击 | `self.contact_forces`, `max_contact_force` | 复用 |

### 2.2 `H1Robot` 子类（`legged_gym/envs/h1/h1.py`）
| 奖励函数 | 作用简介 | 关键依赖 | 对 `g1` 的适配评估 |
| --- | --- | --- | --- |
| `_reward_lin_vel_z` / `_reward_ang_vel_xy` | 同基类，改写以兼容姿态模式 | 同上 | 复用 |
| `_reward_standing` / `_reward_standing_air` | 控制站立/腾空次数 | `standing_envs_mask`, 足端接触力 | 需新增站立场景标记 |
| `_reward_standing_joint_deviation` | 站立时约束上肢 | `self.dof_pos`, `self.default_dof_pos`, 肩肘索引 | `g1` 无上肢，可裁剪掉 |
| `_reward_orientation_control` | 根据指令调节躯干俯仰 | `self.commands[:,8]`, `self.projected_gravity` | 需扩展命令并实现俯仰控制 |
| `_reward_waist_control` | 约束腰部滚转 | `self.commands[:,9]`, `self.torso_inds` | `g1` 无腰关节，可忽略或用于髋关节滚转 |
| `_reward_base_height` | 结合地形扫描保持高度 | `self.heights_below_base`, `self.commands[:,7]` | 需添加足下高度扫描 |
| `_reward_action_rate` | 二阶动作光滑化 | `self.last_last_actions` | 可直接复用 |
| `_reward_dof_vel_limits` | 速度限幅随 `velocity_level` 调节 | 需维护 `self.velocity_level` | 需新增速度等级逻辑 |
| `_reward_feet_contact_forces` | 使用缩放与站立 mask | `self.obs_scales.contact_force`, `standing_envs_mask` | 复用（添加 mask 后） |
| `_reward_collision` / `_reward_termination` / `_reward_dof_pos_limits` / `_reward_torque_limits` | 与基类一致 | -- | 复用 |
| `_reward_tracking_lin_vel` / `_reward_tracking_ang_vel` | 指令跟踪（指数形式） | 同基类 | 复用 |
| `_reward_feet_stumble` | 惩罚足端水平冲击 | `self.contact_forces` | 复用 |
| `_reward_stand_still` | 与基类类似 | `self.commands`, `self.default_dof_pos` | 复用 |
| `_reward_joint_power_distribution` | 平衡小腿功率 | `self.torques`, `self.dof_vel`, 指定小腿索引 | `g1` 需确认膝/踝索引，可复用 |
| `_reward_hip_deviation` | 控制髋部姿态 | `self.hip_inds` | 复用 |
| `_reward_shoulder_deviation` | 约束手臂 | 肩肘索引 | `g1` 无 -> 移除 |
| `_reward_no_fly` | 防止指令场景下的空中停留/双脚离地 | `self.commands`, 接触计数 | 跳跃任务需重新设逻辑（允许部分场景） |
| `_reward_tracking_contacts_shaped_force` / `_reward_tracking_contacts_shaped_vel` | 根据期望接触图约束力和速度 | `self.desired_contact_states`, 足端力与速度 | 需实现期望接触与摆动规划（跳跃核心） |
| `_reward_feet_clearance_cmd_linear` / `_reward_feet_clearance_cmd_polynomial` | 根据命令的摆动高度调整足端高度 | `self.commands[:,6]`, `self.foot_pos_world`, `self.measured_foot_scan`, `self.desired_contact_states` | 需移植足端轨迹规划与扫描 |
| `_reward_feet_slip` | 惩罚足端接触时的水平滑动 | `self.foot_velocity_world`, `self.bool_foot_contact` | 可复用 |
| `_reward_hopping_symmetry` | 保持左右足步态对称 | `self.foot_pos_b_h`, `self.commands[:,4]` | 需添加基坐标中的足端位置 |
| `_reward_alive` | 常数奖励 | -- | 复用 |

### 2.3 `H1InterruptRobot` 重载部分（`legged_gym/envs/h1/h1interrupt.py`）
该类将部分奖励拆分为上下肢或独立版本，主要便于仅激活腿部控制时关闭上肢项。移植 `g1` 时可借鉴其方法，将不适用的奖励重写为空或拆分。

## 3. 奖励迁移策略与优先级
- **直接复用（无需结构调整）**：`lin_vel_z`、`ang_vel_xy`、`torques`、`dof_vel`、`dof_acc`、`action_rate`、`collision`、`termination`、`dof_pos_limits`、`tracking_lin_vel`、`tracking_ang_vel`、`feet_stumble`、`stand_still`、`feet_slip`、`joint_power_distribution`、`hip_deviation`、`alive` 等。需确保 `g1` 观测已输出所需状态。
- **需要新增状态或逻辑**：
  - 脚本计划相关：`tracking_contacts_shaped_*`、`feet_clearance_cmd_*`、`no_fly`、`hopping_symmetry`。要求实现期望接触图、足端在身体坐标系的位置与高度扫描。
  - 模式控制相关：`standing`、`standing_air`、`base_height` 新版、`orientation_control`。需扩展命令并在环境中维护站立/跳跃模式标志。
  - 速度限制：`dof_vel_limits` 依赖 `self.velocity_level`，可根据命令速度或课程策略动态设置。
- **不适用/需移除**：上肢相关（`standing_joint_deviation`、`shoulder_deviation`、`waist_control`）因 `g1` 仅保留下肢，可直接删除或重定向到腿部关节。

## 4. `g1` 跳跃与单脚小跳实现建议
1. **命令层扩展**：
   - 在 `G1HugWBCCfg.commands.ranges` 中加入 `gait_frequency`、`foot_swing_height`、`body_height`、`body_pitch` 等字段，允许策略接收跳跃高度与步态模式（0=单脚跳、0.5=慢走、1=跳跃）。
   - `task_registry` 生成命令时，根据课程阶段混合输出“行走-跳跃-单脚跳”三种模式，实现平滑过渡。
2. **接触调度与腿相位**：
   - 迁移 `H1` 的 `self.desired_contact_states` 与多项式足端规划器 `_polynomial_planer`，针对 `g1` 仅有 2 个足端的情况简化为双状态（左/右）。
   - 在 `_post_physics_step_callback` 中根据当前模式设定 `self.foot_indices` 相位（如跳跃时双腿同相，单脚小跳时交替提升一条腿）。
3. **观测与特权信息**：
   - 增加足底到地面的高度扫描（`self.measured_foot_scan`）与 `self.foot_pos_b_h`，用于奖励 `feet_clearance_cmd_*` 与 `hopping_symmetry`。
   - 在特权观测中加入摩擦系数、接触力、姿态误差，便于 `MlpAdaptModel` 学习落地冲击调节。
4. **奖励组合**：
   - 跳跃阶段着重启用 `feet_clearance_cmd_linear`、`tracking_contacts_shaped_force/vel`、`hopping_symmetry`，并降低 `feet_contact_forces` 的权重以允许短暂高冲击。
   - 单脚小跳阶段保留 `no_fly` 但允许目标腿在指令节律内脱离地面，需根据模式重写奖励掩码。
5. **课程设计**：
   - 先在平地上训练加速度响应与空中保持，再逐步引入高度需求、随机推力与地形扰动。
   - 使用 `curriculum_reward_list` 将新的奖励项纳入课程门槛，避免训练初期因空中惩罚导致策略崩溃。

## 5. 后续工作清单
- [ ] 创建 `references/envs/g1/g1_hugwbc_env.py`，实现 `BaseTask` 架构与新的命令/观测接口。
- [ ] 新增 `G1HugWBCCfg` / `G1HugWBCCfgPPO`，并在 `task_registry` 注册 `g1_hugwbc` 任务。
- [ ] 在 `rsl_rl/algorithms/ppo.py` 增加环境级对称映射配置，写入 `g1` 对应的 `12×12` / 观测置换矩阵。
- [ ] 迁移奖励函数（按照上文分类），并为 `g1` 定义缺失的状态量（`desired_contact_states`、`velocity_level` 等）。
- [ ] 编写最小化冒烟测试脚本：跳跃命令下运行 `2–3k iteration`，验证奖励曲线上升、足端轨迹合理。

通过上述步骤，可以在复用 `HugWBC` 成熟策略架构的同时，专注于 `g1` 下肢控制的目标性能（持续跳跃与交替单脚跳），并为后续实机迁移或仿真扰动实验奠定基础。
