# TinyLLMRL

单轮 LLM 强化学习练习。当前支持批量 rollout 和 Simple Baseline REINFORCE。

| 文件 | 用途 |
| --- | --- |
| `algorithms/reinforce_simple_baseline.py` | REINFORCE with a Simple Baseline |
| `algorithms/reinforce_learned_value_baseline.py` | REINFORCE with a Learned Value Baseline，待实现 |
| `datasets/dataset.py` | 读取、校验和写入 JSONL；参考答案只用于判分 |
| `models/policy.py` | 从服务器本地权重加载模型和 tokenizer |

## 模型下载

首个实验模型用 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。
以下命令只在训练服务器上执行。先激活项目的 `.venv`；如果没有 `hf` 命令，在项目目录用 `uv pip install --python .venv/bin/python huggingface_hub`（本机项目级 uv 路径为 `.tools/uv`）。

```bash
mkdir -p "$HOME/TinyLLMRL/models/weights"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download Qwen/Qwen3-0.6B \
  --local-dir "$HOME/TinyLLMRL/models/weights/Qwen3-0.6B"
```

后续从该目录加载模型和 tokenizer，而不是在训练时重新通过模型 ID 下载。下载方式见 [HF-Mirror](https://hf-mirror.com/) 和 [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。
训练和评测都在 `tokenizer.apply_chat_template` 显式传入 `enable_thinking=False`。

服务器项目根目录下，用 `.venv/bin/python` 导入 `from models.policy import load_policy`，调用 `model, tokenizer = load_policy()`。默认是 CUDA 和 FP32；可通过 `device` 和 `dtype` 显式调整。

服务器环境使用 PyTorch 2.7.1（CUDA 12.8）和 Transformers 4.57.3。后续用 uv 管理项目的 `.venv`，不向系统 Python 安装依赖。

## 外部评测

[SVAMP 原始数据](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json) 放在服务器的 `~/TinyLLMRL/data/raw/svamp/SVAMP.json`。题目由 `Body` 和 `Question` 组成，只用 `Answer` 判分。SVAMP 不用于训练或调参。

## 合成训练数据

在服务器项目根目录运行：

```bash
.venv/bin/python -m datasets.generate_arithmetic --seed 42 --output data/processed/arithmetic-v1
```

生成器先用 seed 打乱所有合法的操作数对，再划分数据，然后把数字填进英文题目模板。加法答案是 `a + b`，减法答案是 `a - b`；一些模板加入不参与计算的数字。只有 `prompt` 进模型，`ground_truth` 留给评分。

- `train.jsonl`：800 题，用于训练。
- `validation.jsonl`：160 题，与训练集共享题型但不共享操作数对，用于开发。
- `template_test.jsonl`：160 题，题型和操作数对均留出，用于最终评测。
- `manifest.json`：seed、Python 版本、源码和数据文件的 SHA-256。复跑时指定新的输出目录，生成器不会覆盖已有目录。

## Rollout 复现约定

数据 seed 和 rollout seed 分开记录。固定权重和 tokenizer 版本、数据哈希、题目及 batch 顺序、生成配置以及 GPU/软件版本；两种算法使用相同的预算和种子规则。生成前用 `model.eval()` 关闭 dropout，使用 `torch.inference_mode()`，在独立复查进程中设置 `torch.manual_seed(rollout_seed)`，不要在同一批次内对每个回答重置成同一个 seed。排查时可在进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并启用 `torch.use_deterministic_algorithms(True)`。

先在同一台服务器上用两个新进程重放同一小批次，逐 token 比较生成结果，而不是只看最终答案。保存 prompt ID、实际生成的 token IDs、采样设置和 seed；如果从训练断点继续，还要保存和恢复 RNG 状态。相同 seed 不保证跨 GPU 或框架版本一致，也不意味着更新后的两个不同策略应生成相同轨迹。

## 服务器目录

以 `~/TinyLLMRL` 为项目根目录。原始数据放在 `data/raw/`，处理和划分后的数据放在 `data/processed/`，模型权重放在 `models/weights/`，实验日志、指标和检查点放在 `outputs/<run-id>/`。这些目录不进 Git；训练和实际数据准备都只在服务器进行。

数据工具可用 Python 3.10+ 单独测试；完整测试在服务器运行 `.venv/bin/python -m unittest discover -s tests -v`。

## Simple Baseline 与 SwanLab

在服务器项目根目录运行（下面是首轮建议配置，不是已完成实验）：

```bash
.tools/uv pip install --python .venv/bin/python 'swanlab[dashboard]'
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --run-name simple-v1 --steps 100
```

默认实时上传指标到 SwanLab。启动时优先读取环境中的 `SWANLAB_API_KEY`，否则读取服务器项目的 `.env`，不需要手动导出。只想写本地看板时用 `--swanlab-mode local`。不要把 `.env` 提交或传到其他机器。配置、逐步指标、SwanLab 日志和最终模型仍留在服务器 `outputs/<run-name>/`；训练脚本不上传模型权重。

- 对照训练奖励和固定验证题的采样正确率。两者都用 `reward.py` 的同一题目后缀和判分规则：生成必须遇到 EOS，且必须只有一条 `FINAL:` 声明，位于最后一行。接受单个整数或可重算验证的单步整数加减等式；最终数值正确才给 1。不评价解释过程，也不用于 SVAMP；新旧奖励的分数不可直接对比。
- 梯度范数和 batch 内优势标准差是诊断信号，不是梯度方差。比较两个 baseline 的方差时，应冻结同一策略、用相同轨迹并按重复 batch 比较梯度估计。
- 间隔记录同一批回答更新前后的序列 log-prob 变化，以及概率比落在 `[0.8, 1.2]` 外的比例。它是观察更新幅度的探针，不是 PPO 的 token 级裁剪比例；REINFORCE 不裁剪，也不复用旧 rollout。

## 双卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc-per-node=2 \
  train.py --run-name simple-ddp-v1 --steps 100 --batch-size 4
```

`--batch-size` 是全局题数。例子中每张卡各生成两条回答，反向时平均梯度；历史 baseline 用全局平均奖励更新。只有 rank 0 评测、写日志和保存模型。比较单卡和双卡时固定全局 batch 和数据 seed，训练速度看 `perf/step_seconds` 和 `perf/generated_tokens_per_second`；这些指标不含验证或保存，也不是纯生成速度。卡数会改变采样流和 batch 形状，不保证轨迹相同。短测试可加 `--no-save`，不用 SwanLab 时可加 `--swanlab-mode disabled`。

服务器项目的 `.venv` 默认加载项目内的 NVIDIA NCCL 2.28.9，不需要 `LD_PRELOAD`。双卡和四卡均通过了 64 MiB 通信和小模型 DDP 梯度测试；这不是 Qwen 的训练测速结果。这里只替换了运行库，包元数据仍是 PyTorch 原来的依赖版本；重建虚拟环境或重新安装 PyTorch/NCCL 后，应重新检查实际加载的 NCCL 版本。
