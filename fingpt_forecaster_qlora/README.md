# FinGPT-Forecaster 4-bit QLoRA 微调管线

本目录为 AStockCrocodile 新增一套独立的可微调大模型管线，技术路线是：

1. FinGPT-Forecaster 思路：把股票历史 K 线、周线、新闻与未来短期表现整理为金融预测指令样本。
2. 4-bit 量化：用 `bitsandbytes` 的 NF4 量化加载 7B 基座模型，降低显存占用。
3. QLoRA 微调：冻结 4-bit 基座，仅训练 LoRA adapter。

公开 FinGPT 项目说明 FinGPT-Forecaster 是面向金融预测/robo-advisor 的组件；HuggingFace 上的 [`FinGPT/fingpt-forecaster_dow30_llama2-7b_lora`](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora) 是 PEFT LoRA adapter，base model 是 Llama-2-7B-chat 系列。FinGPT 项目源码见 [`AI4Finance-Foundation/FinGPT`](https://github.com/AI4Finance-Foundation/FinGPT)。因此本目录默认采用“Llama-2-7B-chat 基座 + FinGPT-Forecaster adapter 初始化 + 本项目 A 股数据继续 QLoRA”的做法。

## 目录结构

```text
fingpt_forecaster_qlora/
  README.md
  requirements.txt
  config.example.env
  common.py
  build_dataset.py
  train_qlora.py
  evaluate.py
  predict.py
  scripts/
    one_click_deploy.sh
    one_click_deploy.ps1
```

## 输入数据

训练数据从本项目 MySQL 数据库读取：

- `dkandles`：日线，要求前复权数据，默认取选股日及之前 55 根。
- `wkandles`：周线，默认取选股日及之前 55 根。
- `klinestatistics`：正样本，默认 `StatType=short_term_surge_3d_20pct`。
- `news`：选股日前后新闻，作为 Forecaster 的市场事件输入。

数据库连接默认读取项目根目录的 `env.txt`。也可以复制本目录的 `config.example.env` 为 `config.env` 后覆盖模型和训练参数。

## 推荐环境

QLoRA 强烈建议在 WSL2/Linux 下跑。Windows 原生可以生成数据集，但 `bitsandbytes` 的 4-bit CUDA 训练通常不可用。

RTX 3060 12GB 建议：

- Ubuntu 22.04 WSL2
- NVIDIA Windows 驱动支持 WSL CUDA
- Python 3.10
- CUDA PyTorch `cu121`
- `max_seq_length=4096`
- `batch_size=1`
- `gradient_accumulation_steps=8`
- `lora_r=16`

如果显存不足，把 `MAX_SEQ_LENGTH` 改成 `3072` 或 `2048`。

## 一键部署

在 WSL2/Linux 项目根目录执行：

```bash
bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh smoke
```

`smoke` 模式会：

1. 创建 `.venv-fingpt-linux`。
2. 安装 PyTorch 和 QLoRA 依赖。
3. 从 MySQL 抽取少量样本生成 JSONL。
4. 加载 4-bit 基座模型和 FinGPT-Forecaster adapter。
5. 运行短训练。
6. 用验证集跑一次评估。

正式训练：

```bash
bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh full
```

Windows 原生只建议先生成数据集：

```powershell
.\fingpt_forecaster_qlora\scripts\one_click_deploy.ps1 smoke
```

如果系统没有 `py.exe`，脚本会自动尝试使用项目已有的 `.\.venv\Scripts\python.exe`、`python` 或 `python3`。也可以手动指定：

```powershell
$env:PYTHON_BIN='D:\Documents\StockInfoCrawler\.venv\Scripts\python.exe'
.\fingpt_forecaster_qlora\scripts\one_click_deploy.ps1 smoke
```

脚本最后会提示切到 WSL2/Linux 执行真正 4-bit QLoRA。

WSL2/Linux 脚本默认使用独立虚拟环境 `.venv-fingpt-linux`，避免和 Windows 原生脚本创建的 `.venv-fingpt` 冲突。需要自定义路径时：

```bash
VENV_DIR=.venv-fingpt-linux bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh smoke
```

## 分步命令

生成训练集：

```bash
python -m fingpt_forecaster_qlora.build_dataset \
  --start-date 20100101 \
  --end-date 20251231 \
  --positive-limit 2000 \
  --negative-ratio 1.0 \
  --valid-ratio 0.2 \
  --daily-window 55 \
  --weekly-window 55 \
  --min-success-rate 0.40
```

训练 QLoRA：

```bash
python -m fingpt_forecaster_qlora.train_qlora \
  --base-model NousResearch/Llama-2-7b-chat-hf \
  --forecaster-adapter FinGPT/fingpt-forecaster_dow30_llama2-7b_lora \
  --data-dir fingpt_forecaster_qlora/data \
  --output-dir fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora \
  --max-seq-length 4096 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

评估：

```bash
python -m fingpt_forecaster_qlora.evaluate \
  --adapter-dir fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora/adapter \
  --threshold 0.40 \
  --max-samples 500
```

单只股票推理：

```bash
python -m fingpt_forecaster_qlora.predict \
  --scode 000001 \
  --trade-date 20260512 \
  --adapter-dir fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora/adapter
```

模型会输出 JSON：

```json
{
  "label": "positive",
  "success_probability": 0.42,
  "confidence": 0.66,
  "key_patterns": ["..."],
  "risk_factors": ["..."]
}
```

`success_probability >= 0.40` 时，才视为满足当前最低成功率门限。

## 配置项

复制配置模板：

```bash
cp fingpt_forecaster_qlora/config.example.env fingpt_forecaster_qlora/config.env
```

常用项：

- `BASE_MODEL`：基座模型。默认 `NousResearch/Llama-2-7b-chat-hf`，也可以填本地路径。
- `FINGPT_FORECASTER_ADAPTER`：FinGPT-Forecaster 初始 adapter。
- `OUTPUT_DIR`：训练输出目录。
- `DATA_DIR`：JSONL 数据集目录。
- `MIN_SUCCESS_RATE`：最低成功率门限，默认 `0.40`。
- `MAX_SEQ_LENGTH`：最大上下文长度，显存不够时降低。

## 输出产物

- `fingpt_forecaster_qlora/data/train.jsonl`
- `fingpt_forecaster_qlora/data/valid.jsonl`
- `fingpt_forecaster_qlora/data/all.jsonl`
- `fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora/adapter/`

`adapter/` 就是训练后的 LoRA 权重，可用于后续选股推理。

## 注意事项

- 这不是直接“训练一个全参数 7B 模型”，而是 QLoRA adapter 微调，适合 RTX 3060 级别显卡。
- 如果使用 `meta-llama/Llama-2-7b-chat-hf`，需要 HuggingFace 账号有 Llama 2 授权；默认示例使用 `NousResearch/Llama-2-7b-chat-hf` 以减少授权阻碍。
- FinGPT-Forecaster adapter 没有 tokenizer，tokenizer 应来自基座模型。
- 训练结果只能作为量化研究信号，不能作为投资建议。
