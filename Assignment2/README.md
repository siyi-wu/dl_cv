# 从零训练迷你扩散模型（MNIST DDPM）

本项目完成第 2 讲课后编程实验：实现 DDPM 前向加噪、小型 U-Net 噪声预测器、DDPM ancestral sampling，并对 1000 / 200 / 50 个反向步的生成效果与耗时进行对比。项目还实现了确定性 DDIM（`eta=0`）作为选做扩展。

## 快速运行教程（Windows）

### 只查看已经完成的实验

无需安装 Python、Conda 或 CUDA，直接双击项目根目录下的 `start_demo.bat`。默认浏览器会打开离线实验展示页；请保持 `demo/` 和 `outputs/` 目录位于项目中。

### 从头训练并生成对比结果

在 Anaconda PowerShell Prompt 中进入项目目录，然后依次执行：

```powershell
conda create -n assignment2-ddpm python=3.11 pip -y
conda activate assignment2-ddpm
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m unittest discover -s tests -v
python train.py --data-dir data --output-dir outputs/train_full --epochs 20 --batch-size 128 --device cuda --num-workers 4
python compare_steps.py --checkpoint outputs/train_full/checkpoint_last.pt --output-dir outputs/comparison --steps 1000 200 50 --num-samples 64 --device cuda
```

训练结束后，双击 `start_demo.bat` 查看损失曲线、样本网格和步数对比。首次训练会自动下载 MNIST；W&B 是可选功能，以上命令无需登录即可运行。若没有 NVIDIA GPU，将训练命令和对比命令中的 `--device cuda` 改为 `--device cpu`，但运行时间会明显增加。

## 1. 工程结构

```text
Assignment2/
├── src/miniddpm/
│   ├── data.py          # MNIST 下载与 IDX 解析（不依赖 torchvision）
│   ├── diffusion.py     # 噪声调度、q(x_t|x_0)、DDPM/DDIM 采样
│   ├── model.py         # < 5M 参数的迷你 U-Net
│   └── utils.py         # 随机种子、设备、图像网格、统计量
├── tests/test_core.py   # 核心公式、形状、参数量、采样测试
├── train.py             # 训练入口，保存原始权重和 EMA 权重
├── sample.py            # 单次生成入口
├── compare_steps.py     # 1000/200/50 步质量代理指标与耗时对比
├── scripts/md_to_pdf.py # 将实验报告 Markdown 转为 PDF
├── report/              # 实验报告 Markdown 与 PDF
├── demo/index.html      # 离线交互实验展示页
├── start_demo.bat       # Windows 一键打开离线展示页
├── outputs/             # 已保留的训练曲线、样本网格和对比数据
└── requirements.txt
```

## 2. 方法依据

实现以 Ho、Jain 与 Abbeel 的 [DDPM 原始论文](https://arxiv.org/abs/2006.11239)为基础，采用噪声预测 MSE 与 U-Net；参考 Nichol 与 Dhariwal 的 [Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html)加入更适合低分辨率图像的余弦调度；参考 Song、Meng 与 Ermon 的 [DDIM](https://arxiv.org/abs/2010.02502)实现无需重新训练的加速采样。数据规模与 IDX 格式以 [MNIST 官方页面](https://yann.lecun.org/exdb/mnist/index.html)为准。

## 3. Conda 环境与依赖

- Windows 10/11（Linux/macOS 也可）
- Conda 24+
- Python 3.11
- PyTorch 2.5.1；可选 NVIDIA CUDA，没有 GPU 时自动使用 CPU

本仓库已创建独立环境 `assignment2-ddpm`。目标机器是 NVIDIA GeForce RTX 4050 Laptop GPU（6 GB），驱动支持 CUDA 13.3；项目使用驱动向后兼容的 PyTorch CUDA 12.4 官方 wheel，不要求本机另装 `nvcc`。创建方式：

```powershell
conda create -n assignment2-ddpm python=3.11 pip -y
conda run -n assignment2-ddpm python -m pip install -r requirements.txt
```

日常使用：

```powershell
conda activate assignment2-ddpm
```

`requirements.txt` 已通过 PyTorch 官方 CUDA 12.4 wheel 索引锁定 `torch==2.5.1+cu124`。安装后必须看到 `True` 和显卡名称：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

当前实现不依赖 `torchvision`。MNIST 下载、解析和图片网格写出均由项目自身完成，可规避 PyTorch/torchvision 二进制版本不匹配问题。完整直接依赖已写入 `requirements.txt`。PyTorch 2.5.1 的 CUDA 12.4 安装选择来自[官方历史版本安装页](https://pytorch.org/get-started/previous-versions/)。

## 4. 数据

首次训练会从 PyTorch 官方 MNIST 镜像下载 4 个压缩 IDX 文件到 `data/MNIST/raw/`，随后自动进行 MD5 校验并载入。训练图像被转换为 `float32` 张量，并从 `[0, 1]` 线性归一化到 `[-1, 1]`。

如需离线运行，可预先将下列文件放入 `data/MNIST/raw/`：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

## 5. 训练

正式训练（默认 1000 个扩散时间步、余弦 beta 调度、20 epochs）：

```powershell
python train.py --data-dir data --output-dir outputs/train_full --epochs 20 --batch-size 128 --device cuda --num-workers 4
```

若显存/内存有限，可减小批量；若只需快速检查完整链路：

```powershell
python train.py --data-dir data --output-dir outputs/smoke --base-channels 16 --batch-size 32 --max-train-samples 256 --max-steps 10 --epochs 2
```

常用参数：

- `--schedule {linear,cosine}`：噪声调度，默认 `cosine`；
- `--timesteps 1000`：训练扩散步数；
- `--base-channels 32`：U-Net 基础通道数，启动时强制检查参数量不超过 5M；
- `--max-train-samples N` / `--max-steps N`：可复现的小规模调试；
- `--device auto|cpu|cuda`：运行设备；
- `--seed 42`：Python、NumPy、PyTorch 和 DataLoader 的随机种子。

CUDA 训练默认自动启用混合精度，DataLoader 使用 pinned memory；RTX 4050 6 GB 可从 batch size 128 开始，若出现显存不足则改为 64。训练目录包含 `checkpoint_last.pt`、`loss_curve.png`、`metrics.jsonl` 和 `training_summary.json`，分别保存完整状态、损失曲线、逐 epoch 指标与汇总。每轮结束都会更新 `checkpoint_last.pt`；按 `Ctrl+C` 会额外保存可恢复的 `checkpoint_interrupted.pt`。

### 使用 Weights & Biases 实时观察

1. 在 [W&B](https://wandb.ai/) 注册并从 User Settings 创建 API key；
2. 在激活的 Conda 环境中登录（密钥不要写进代码或 Git）：

```powershell
wandb login
```

3. 启动带在线追踪的 GPU 训练：

```powershell
python train.py --data-dir data --output-dir outputs/train_full --epochs 20 --batch-size 128 --device cuda --num-workers 4 --wandb-mode online --wandb-project mnist-ddpm --wandb-name rtx4050-cosine-20ep
```

终端会打印 run URL。网页会实时显示 `train/loss`、`train/gradient_norm`、学习率、逐轮平均/末次损失和每轮 DDIM 50 步预览图。若还想记录参数与梯度直方图，追加 `--wandb-watch`，但会增加少量开销。网络不稳定时用 `--wandb-mode offline`，之后执行 `wandb sync outputs/train_full/wandb/<离线运行目录>` 上传。项目锁定 `wandb==0.22.3`，可识别 W&B 新版长 API key；更旧 SDK 可能误报“必须为 40 位”。以上流程遵循 [W&B 官方 Quickstart](https://docs.wandb.ai/models/quickstart)和[媒体记录说明](https://docs.wandb.ai/models/track/log/media)。

中断后恢复（模型、EMA、优化器、AMP scaler、global step 和原 W&B run ID 会一起恢复）：

```powershell
python train.py --data-dir data --output-dir outputs/train_full --epochs 20 --batch-size 128 --device cuda --num-workers 4 --wandb-mode online --wandb-project mnist-ddpm --resume outputs/train_full/checkpoint_interrupted.pt
```

恢复时模型结构与扩散参数必须和原命令一致；W&B 使用官方建议的 `resume="allow"` 和保存的 run ID 接续同一曲线。参见 [W&B 恢复运行文档](https://docs.wandb.ai/models/runs/resuming)。

## 6. 采样与测试

用 EMA 权重执行 1000 步 ancestral sampling 并输出 8×8 网格：

```powershell
python sample.py --checkpoint outputs/train_full/checkpoint_last.pt --output outputs/samples/ddpm_1000.png --sampler ancestral --steps 1000 --num-samples 64 --device cuda
```

使用 DDIM 50 步确定性采样（选做）：

```powershell
python sample.py --checkpoint outputs/train_full/checkpoint_last.pt --output outputs/samples/ddim_50.png --sampler ddim --steps 50 --eta 0 --num-samples 64 --device cuda
```

采样图片旁会生成同名 `.json`，记录耗时、采样器、步数、设备、随机种子和无参考图像统计量。运行自动化测试：

```powershell
python -m unittest discover -s tests -v
```

## 7. 1000 / 200 / 50 步对比

```powershell
python compare_steps.py --checkpoint outputs/train_full/checkpoint_last.pt --output-dir outputs/comparison --steps 1000 200 50 --num-samples 64 --device cuda
```

脚本对三组实验复用同一随机初始噪声并预热模型，输出各自 8×8 网格、`step_comparison.png` 并排预览，以及 `benchmark.json` / `benchmark.csv`。表中 sharpness、diversity 与 foreground ratio 是无需额外分类器的可重复质量代理，不能替代人工观察、FID 或分类器置信度。

### 实测结果

| Ancestral 步数 | 总耗时/s | 相对 1000 步加速 | Sharpness | Diversity | 前景占比 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 5.291 | 1.00× | 0.06445 | 0.18198 | 0.12512 |
| 200 | 1.063 | 4.98× | 0.06748 | 0.18190 | 0.13716 |
| 50 | 0.274 | 19.28× | 0.06538 | 0.17960 | 0.12984 |

![1000/200/50 步 ancestral 采样对比](outputs/comparison/step_comparison.png)

三组均能生成清晰且多样的手写数字。200 步与 1000 步视觉差异很小，同时接近 5 倍加速；50 步仍大多可辨识，但少量样本的笔画更粗、形状更含混。三个代理指标变化不大，与视觉结论基本一致。由于 ancestral 过程本身随机，脚本虽共享初始噪声，不同组的逐步随机噪声仍不同，因此不应把单个位置作为严格配对样本。

DDIM 50 步独立运行耗时 0.598 秒；部分数字清晰，但断笔、重影和畸形比 50 步 ancestral 更明显。该耗时包含首次 CUDA kernel 冷启动，不能与已预热的 `compare_steps.py` 时间直接比较。

![DDIM 50 步样本网格](outputs/samples/ddim_50.png)

## 8. 实验报告

报告源文件位于 `report/experiment_report.md`。可重建 PDF：

```powershell
python scripts/md_to_pdf.py --input report/experiment_report.md --output report/experiment_report.pdf
```

转换脚本使用 ReportLab，并自动搜索常见 Windows/Linux 中文字体。当前报告已引用 `outputs/` 中的真实损失曲线与采样图，并完成 PDF 逐页渲染检查。

## 9. 复现约定

### 离线展示页

#### 换电脑一键展示（Windows）

复制或克隆完整的 `Assignment2` 目录到目标电脑，双击项目根目录下的 `start_demo.bat` 即可。脚本会使用系统默认浏览器打开 `demo/index.html`，不需要 Conda、Python、GPU、网络或安装任何依赖。

必须保留以下相对目录结构，否则页面中的实验图片无法加载：

```text
Assignment2/
├── start_demo.bat
├── demo/
│   └── index.html
└── outputs/
    ├── train_full/
    ├── samples/
    └── comparison/
```

也可以直接双击 `demo/index.html`。页面不依赖服务器或 CDN，包含原生数学公式、完整 MiniUNet 架构、章节导航、训练轮次滑块、1000/200/50 步切换、耗时图、DDPM/DDIM 对照和图片放大。

如果希望通过 `localhost` 地址展示，并且目标电脑已经安装 Python，可在项目根目录运行：

```powershell
python -m http.server 8000
```

然后访问 `http://localhost:8000/demo/`。

- 所有命令均从 `Assignment2` 根目录执行；
- 原始数据、训练产物和报告分离，输出目录由命令行明确指定；
- checkpoint 保存完整模型/扩散配置，采样脚本优先使用 EMA 权重；
- 对比实验固定初始噪声与种子，避免样本差异干扰步数比较；
- `git` 忽略下载数据、checkpoint、W&B 本地缓存和临时文件；保留 `outputs/` 下的 PNG、JSON、CSV、JSONL 结果以及报告，便于换电脑直接展示。
- 另一台电脑只做展示时无需 checkpoint：克隆仓库后可直接打开 README、`outputs/` 图片和 `report/experiment_report.pdf`。如需重新采样或继续训练，需另行复制被忽略的 checkpoint。

## 10. 本次完整实验结果

- 硬件：NVIDIA GeForce RTX 4050 Laptop GPU（6 GB）；
- 模型：1,155,265 参数，满足不超过 5M 的限制；
- 训练：MNIST 60,000 张、20 epochs、9,380 个优化步，总计 361.85 秒；
- 损失：epoch 平均 MSE 从 0.11124 降至 0.03945，最终 batch MSE 为 0.03696；
- 输出：训练损失曲线、20 轮预览、DDPM/DDIM 8×8 网格、1000/200/50 对比图及 JSON/CSV 均已生成；
- 验证：4 项核心单元测试全部通过，W&B run ID 为 `70zndm3y`。

![完整训练损失曲线](outputs/train_full/loss_curve.png)

`outputs/train_full/checkpoint_last.pt` 被 `.gitignore` 排除，以避免向 Git 提交约 18 MB 的权重；其余实验结果不再被忽略，可以在另一台电脑上直接展示。
