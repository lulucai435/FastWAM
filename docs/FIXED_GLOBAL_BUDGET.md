# FastWAM 固定全局动作计算预算

开启固定预算后，每次 plan 使用一张 10×30 执行表。10 个去噪步都执行。表的行表示去噪步，列表示动作 block，`true` 表示复用。重算时缓存本层的 `h_out - h_in`，复用时把缓存残差加到当前输入。每个 plan 重新建立残差缓存，视觉前缀也从当前观测重新计算。

| `EVALUATION.fixed_action_schedule` | 每 plan 重算数 | 来源 |
|---|---:|---|
| `null` | 原版设置 | 原版推理入口 |
| `full300` | 300 | 10 步全算，可使用与低预算相同的编译后端 |
| `global50` | 165 | 原 ACS 的 50% 跳过档，比例以首步之外的 270 个位置为分母 |
| `terminal97` | 97 | 固定全局总额，联合优化各层次数和重算位置。每步重算数为 `[30,3,1,1,1,3,22,5,1,30]` |
| JSON 文件路径 | 按表 | 用下方工具求出的自定义表 |

`EVALUATION.fixed_action_backend=graph` 使用 torch.compile 和 CUDA Graph。第一次遇到新输入形状时会热身。选择 `eager` 可执行同一张表以便调试。固定表支持 30 层的 FastWAM 动作推理路径，去噪步数设为 10。

## 环境和资源

以 [FastWAM 环境安装](../README.md#environment-setup) 和 [RoboTwin 评估](../README.md#inference-with-released-checkpoints) 为基础。历史实验使用 Python 3.10、PyTorch 2.7.1+cu128、TorchVision 0.22.1+cu128、H100 80 GB、NVIDIA 580.95.05、CUDA 12.8、cuDNN 9.7.1 和 BF16。求表验证用 SciPy 1.15.3 的 HiGHS MILP。

```bash
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[schedule]'
python scripts/acceleration/preflight.py
python scripts/acceleration/preflight.py --cuda
```

仿真需按 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 安装机器人、物体和任务配置。历史运行用 SAPIEN 3.0.0b1、mplib 0.2.1、cuRobo 0.7.8 和带 libx264 的 FFmpeg，评估入口用仓库自带的 `third_party/RoboTwin/script/eval_policy.py`。如果使用外部 RoboTwin 根目录，可用 `EVALUATION.robotwin_root` 指向该目录，使用本分支的评估入口以生成逐条预算统计。原生 `task_config` 目录需要提供 `demo_clean.yml`、`_camera_config.yml`、`_embodiment_config.yml` 及 `_eval_step_limit.yml`。 实测仿真依赖在单独的 Python 路径中加载，SAPIEN 等旧二进制包与 NumPy 1.26.4 一起运行。遇到 NumPy ABI 不兼容时，用独立仿真环境或仅为评估进程设置 NumPy 1.26.4 的模块搜索路径。求表工具可以在仓库的常规环境中运行。

在 GPU 容器中启用 `NVIDIA_DRIVER_CAPABILITIES=all`，确认 `nvidia-smi` 和 CUDA 预检，再从 RoboTwin 目录运行 `python script/test_render.py` 检查离屏渲染。出现 CUDA 803 时，将宿主驱动库目录 `/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu` 放在 CUDA compat 库之前。离屏相机还需要与宿主驱动匹配的 Vulkan/EGL 库。

若 CUDA 能初始化，但预检报告加载了旧的 cuDNN（本机曾加载到 9.3），让当前 PyTorch 环境的 cuDNN 目录优先于系统目录。以下在 Bash 中运行，路径填写当前 PyTorch 环境中实际的 `nvidia/cudnn/lib`：

```bash
CUDNN_LIB=/path/to/venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="$CUDNN_LIB:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cudnn_files=("$CUDNN_LIB"/libcudnn*.so.9)
printf -v cudnn_preload '%s:' "${cudnn_files[@]}"
export LD_PRELOAD="${cudnn_preload%:}${LD_PRELOAD:+:$LD_PRELOAD}"
python scripts/acceleration/preflight.py --cuda
```

历史设置下预检应显示 `cudnn: 90701`。

从 [FastWAM release](https://huggingface.co/yuanty/fastwam) 获取资源。模型加载需要 Wan2.2 的 VAE、文本编码器和 tokenizer，可以让 loader 下载，也可用 `DIFFSYNTH_MODEL_BASE_PATH` 指向本地模型目录。全部资源就绪后可设置 `DIFFSYNTH_SKIP_DOWNLOAD=true`。

| 文件 | 历史实验 SHA256 |
|---|---|
| `robotwin_uncond_3cam_384.pt` | `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` |
| `robotwin_uncond_3cam_384_dataset_stats.json` | `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` |

## 跑一条 RoboTwin 固定 97 仿真

在仓库根目录运行，把两个路径换成实际资源路径。每次预测 32 步、执行 8 步后重新规划，去噪 10 步，sigma shift 为 5。

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_alarmclock EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=1 EVALUATION.action_horizon=32 \
  EVALUATION.replan_steps=8 EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5 EVALUATION.fixed_action_schedule=terminal97 \
  EVALUATION.fixed_action_backend=graph EVALUATION.skip_get_obs_within_replan=false \
  mixed_precision=bf16 seed=42
```

改用 `global50` 或 `full300` 可做同机对照，每个分支使用独立输出目录。仿真结果目录里有原生 SR、视频及 `fixed_budget_episodes.jsonl`。JSONL 每行保存任务、物理 seed、成败、停止原因、执行动作数、plan 数、执行表哈希、每 plan 推理毫秒和整条轨迹的重算数。首个 plan 可能包含编译和暖机，捕获时间另记。实际仿真时间还包括渲染、写盘和完成任务所需的动作数。

## 求一张新的表

在独立的校准观测上完整去噪，记录每层每步的 `h_out - h_in`。将残差展平，计算同一层各去噪步之间的两两余弦相似度，再跨样本平均。将结果存成 `[30,10,10]` 的 `.npy`，轴依次为动作层、缓存来源步和当前步。旧 terminal97 使用 12 个任务各一条轨迹，每条抽 3 个状态。

```bash
python scripts/acceleration/solve_schedule.py \
  --similarity /path/to/similarity.npy --budget 165 --name terminal165 \
  --output ./terminal165.json
```

用 `EVALUATION.fixed_action_schedule=/absolute/path/to/terminal165.json` 执行新表。默认约束为首末步各重算 30 层、第二步重算前三层、每步至少一层，最低可行预算为 70。可用 `--no-terminal-anchor` 和 `--allow-empty-step` 改变约束。MILP 在给定预算下最大化缓存残差相似度，求解证书和输入哈希保存在 JSON。SR 和时延需要配对实验。

冻结的 97、165 表经 `preflight.py` 校验。使用原实验相似度矩阵重新求解 97，应得到冻结表的同一 mask。分支内提供的是冻结执行表。新任务的相似度矩阵由接手方用自己的校准数据生成。


## 历史参考

原 12 个 RoboTwin 任务各 10 个场景的同场景对照中，固定 97 成功 102/120，原 ACS 165 成功 108/120，原版 FastWAM 成功 105/120。97 和 165 的历史平均规划耗时分别为 34.68 ms 和 42.28 ms。这些数字来自旧实验，用于校对复现协议和测量范围。用本分支重新比较性能时，固定 checkpoint、种子、初始场景和计时口径，并将首轮热身与稳态推理分开。
