# 训练与推理一键命令

请在 `Assignment2` 项目根目录中运行以下命令。命令直接使用已经创建的 Conda 环境 `assignment2-ddpm`，不需要提前执行 `conda activate`。

## 训练一键运行命令

下面一行命令会先训练实验一 MiniUNet，再训练实验二 EnhancedUNet；两个模型均使用 GPU、20 个 epoch 和 batch size 128。

```bash
conda run -n assignment2-ddpm python train.py --model mini --data-dir data --output-dir outputs/train_full --epochs 20 --batch-size 128 --device cuda --num-workers 4 && conda run -n assignment2-ddpm python train.py --model enhanced --data-dir data --output-dir outputs/train_enhanced --epochs 20 --batch-size 128 --device cuda --num-workers 4
```

训练完成后的 checkpoint：

- MiniUNet：`outputs/train_full/checkpoint_last.pt`
- EnhancedUNet：`outputs/train_enhanced/checkpoint_last.pt`

## 推理一键运行命令

下面一行命令会使用两个模型的 EMA 权重，各生成 64 张 50 步 ancestral 样本，并写入 `outputs/inference/`。

```bash
conda run -n assignment2-ddpm python sample.py --checkpoint outputs/train_full/checkpoint_last.pt --output outputs/inference/mini_ancestral_50.png --sampler ancestral --steps 50 --num-samples 64 --seed 123 --device cuda && conda run -n assignment2-ddpm python sample.py --checkpoint outputs/train_enhanced/checkpoint_last.pt --output outputs/inference/enhanced_ancestral_50.png --sampler ancestral --steps 50 --num-samples 64 --seed 123 --device cuda
```

推理完成后的图片：

- `outputs/inference/mini_ancestral_50.png`
- `outputs/inference/enhanced_ancestral_50.png`

每张图片旁边还会自动生成同名 JSON 文件，记录模型、设备、采样步数、耗时和统计指标。
