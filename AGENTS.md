# Repository Guidelines

## 项目结构与模块组织
主要仿真任务位于 `legged_gym/envs`，其中 `legged_gym/envs/h1` 隔离 Humanoid 配置，`legged_gym/envs/base` 提供可复用控制逻辑。通用工具集中在 `legged_gym/utils` 与 `legged_gym/legged_utils`，调参脚本保存在 `legged_gym/scripts`。强化学习基元驻留 `rsl_rl/rsl_rl`，同目录下的 `setup.py` 用于可编辑安装；`references/` 复制主要目录结构以存放对照实验。机器人网格、URDF 与贴图放在 `resources/robots`，新模型需包含 README 说明尺度与坐标系。提交前确认新增资源不会覆盖现有资产，并在 `CHANGELOG` 或 PR 描述标记依赖关系。Key directories: `legged_gym/envs`, `rsl_rl/rsl_rl`, `resources/robots`.

## 构建、测试与开发命令
首次部署：`conda create -n hugwbc python=3.8`、`conda activate hugwbc`，随后执行 `pip install -e rsl_rl` 与 `pip install -e legged_gym`（在根目录运行）以挂载本地包。策略训练命令 `python legged_gym/scripts/train.py --task=h1int --headless` 支持追加 `--max_iterations` 限定步数；`python legged_gym/scripts/play.py --task=h1int --record` 可输出视频帧。调试外部依赖时使用 `python -m pip list` 验证版本，必要时将环境写入 `environment.yml`：`conda env export > environment.yml`。若需释放 GPU，可通过 `CUDA_VISIBLE_DEVICES=0 python ...` 控制显卡。Useful flow: create env → editable installs → train → play → export configs.

## 代码风格与命名约定
Python 全面使用 4 空格缩进，模块与函数保持 snake_case，类与配置对象使用 CapWords。常量以全大写或 UPPER_SNAKE_CASE 命名，并把路径、超参集中到 `cfg` 对象中。公共 API 推荐补全 `typing` 注解，复杂 Tensor 运算前加行内注释解释形状。提交前运行 `python -m compileall legged_gym rsl_rl references`，若引入格式化工具请在 PR 说明里写明 `ruff` 或 `black` 版本并限制差异范围。Notebook、可视化脚本应放入 `references/scripts`，文件名遵循 `<theme>_<purpose>.py`。Style reminder: 4-space indent, snake_case functions, CapWords classes, documented tensors.

## 测试指引
当前未提供自动化基准，请在提交前运行训练脚本至少 3k iterations，记录 tensorboard 或终端奖励曲线并附在 PR。针对确定性逻辑可在根目录创建 `tests/`，使用 `pytest -q` 或 `python -m pytest tests/test_xxx.py` 触发用例；若需要模拟 Isaac Gym 接口，可引入轻量 Mock 并解释假设。评估学到的策略时建议保存 `output/ckpt_last.pt` 与日志，以便他人复现实验。破坏兼容性的改动需在同一 PR 内提供回滚方案或迁移指南。Testing checklist: run training smoke test, add pytest cases for utilities, attach logs.

## 提交与合并请求规范
Commit 信息采用祈使句加范围，如 `Add H1_2 environment`, `Fix play.py latency`，引用 Issue 时写 `ref #ID`。PR 描述包含修改动机、核心变更列表、验证命令与输出摘要，若调整模型或资源请附加截图、短视频或 `wandb` 链接。保持单一主题，拆分环境配置变动与 RL 算法更新。涉及共享控制器、SDK 桥接或资源格式的改动请 @ 项目维护者，并在标题添加 `[core]`、`[sim]` 等前缀方便筛选。PR recipe: imperative commits, linked issues, focused scope, reviewer-friendly evidence.

## 仿真资产与配置提示
新增机器人资产放入 `resources/robots/<model>`，体积超过 10 MB 的文件启用 Git LFS；引用第三方素材需在 PR 中标明来源与许可证。复制配置时优先基于 `legged_gym/envs/h1/config.py` 并仅覆盖差异，避免破坏默认任务。若需要外部硬件常量或 ROS 参数，请记录在 PR 和 `resources/robots/<model>/README.md` 中，方便团队追踪。运行重型仿真前检查 GPU 利用率并使用 `nvidia-smi` 监控。Asset tips: isolate new robot data, document provenance, keep configs minimal and reproducible.
