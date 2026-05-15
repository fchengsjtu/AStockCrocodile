# Qwen2.5 股票模式微调管线

本目录重建为一套独立的本地大模型微调管线，基座模型固定为 `Qwen/Qwen2.5-0.5B-Instruct`。训练样本来自 `klinestatistics`，每个正样本使用 `PrevTradeDate` 作为选股锚点，输入为该日期及之前的 55 根日 K 线和 55 根周 K 线。数据集按稳定哈希切分为 80% 训练集、20% 测试集。

目标：模型对任意股票某个日期的前 55 根日线和 55 根周线进行判定；如果输出 `label=positive`，评估集上的成功率需要不低于 `20%`。

## 一键部署、训练、评估

WSL2/Linux 推荐命令：

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash llm_finetune/scripts/one_click_deploy.sh smoke
```

正式数据量：

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash llm_finetune/scripts/one_click_deploy.sh full
```

Windows 原生命令可用于小规模验证：

```powershell
cd D:\Documents\StockInfoCrawler
.\llm_finetune\scripts\one_click_deploy.ps1 smoke
```

## 分步命令

生成 80/20 数据集：

```bash
python -m llm_finetune.build_dataset \
  --output-dir llm_finetune/data \
  --stat-type short_term_surge_3d_20pct \
  --positive-limit 2000 \
  --negative-ratio 1.0 \
  --daily-window 55 \
  --weekly-window 55 \
  --batch-size 30
```

训练 LoRA/QLoRA：

```bash
python -m llm_finetune.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir llm_finetune/data \
  --output-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4
```

Windows 没有可用 `bitsandbytes` 时加 `--no-4bit`：

```powershell
python -m llm_finetune.train --no-4bit
```

评估并强制成功率不低于 20%：

```bash
python -m llm_finetune.evaluate \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora/adapter \
  --data-dir llm_finetune/data \
  --threshold 0.40 \
  --min-success-rate 0.20 \
  --max-samples 500
```

预测单只股票：

```bash
python -m llm_finetune.predict \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora/adapter \
  --scode 000001 \
  --date 20260512
```

## 自动化测试

```bash
python -m unittest tests.test_llm_finetune -v
```

一键脚本最后会自动执行该测试。评估脚本如果成功率低于 `--min-success-rate` 或没有任何正向推荐，会返回失败退出码。

