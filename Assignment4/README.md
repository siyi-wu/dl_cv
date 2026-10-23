# Assignment4：卷积分类、空间结构先验与扩散生成

本目录根据 `task.md` 完成实验，沿用 Assignment3 的数据读取、固定划分和单隐藏层 MLP 结构。包含可运行代码、四组实测分类结果、DDPM 权重与生成评价、嵌入图片的中文 Markdown 报告（按用户要求仅提供 MD 格式）。原任务说明保持不变。

## 环境与运行

已验证环境：Windows、Python 3.11.16、PyTorch 2.5.1+cu124、NumPy 1.26.4，NVIDIA GeForce RTX 4050 Laptop GPU。详细版本见 `outputs/environment.json`。

```powershell
conda activate assignment2-ddpm
cd E:\Github_Project\dl_cv\Assignment4
python verify.py
python run.py
python make_report.py
```

新环境安装 `pip install -r requirements.txt`，GPU 版 PyTorch 按本机 CUDA 环境安装；CPU 可运行 `python run.py --device cpu`，耗时会明显增加。报告生成仅依赖 Python 标准库。分享报告时请同时保留 outputs 图片目录。

数据默认复用 `../Assignment3/data/FashionMNIST/raw`。独立提交时可把配置中的 `data_dir` 改成 `data/FashionMNIST/raw`，再运行 `python download_data.py` 下载官方数据。所有相对路径均以本目录为基准。

按阶段运行：

```powershell
python run.py --stage classify
python run.py --stage diffusion
python run.py --stage sample
python make_report.py
```

`sample` 需要已存在的 `diffusion.pt` 和 `cnn_original.pt`。默认从配置种子重新构造初始噪声并保存为 `initial_noise.pt`；同环境下可重复生成相同结果。运行会覆盖同名实验输出；需对比新配置时请先备份 outputs。配置改变后，应完整重新运行所有阶段，避免混用不同实验结果。

## 分类比较的控制条件

- 官方 60,000 张训练图用种子 42 的 `torch.randperm` 分成训练 50,000 / 验证 10,000；划分与 Assignment3 一致。官方测试 10,000 张仅最终评价。
- 四组均从头训练 20 epoch、batch 128、SGD、学习率 0.1、训练种子 42；独立批次生成器保证四组训练顺序一致。
- MLP：784 → 128 → ReLU → 10，共 101,770 参数。CNN：两组 3×3 卷积、ReLU、2×2 最大池化，再经 64 维隐藏层分类，共 105,866 参数。
- 原图输入 [0,1]；置换种子 2026，训练、验证、测试共用同一置换。置换保存为 `permutation.npy`，划分索引保存为 `split.json`。
- 按最佳验证准确率选择权重，不依据测试集调参。相同预算指相同更新次数，不指相同墙钟时间。
- 驻 GPU 的批处理流程不同于 Assignment3 的 DataLoader，因此重训 MLP 结果不必与旧报告逐位相同。GPU 确定性卷积已开启，但不承诺跨设备/版本的位级复现。

| 模型 | 原图测试准确率 | 置换图测试准确率 |
|---|---:|---:|
| MLP | 87.23% | 87.29% |
| CNN | 90.26% | 86.91% |

原图 CNN 领先 3.03 个百分点，置换后落后 0.38 个百分点。固定置换保留信息但破坏局部邻域；结果支持 CNN 优势依赖空间结构。仅运行一个训练种子，不能推断跨种子的统计显著性。

训练耗时含训练、验证和 checkpoint 保存，CUDA 前后同步；排除数据读取和绘图。显存指标是整个进程的峰值已分配显存，包含驻留数据和置换副本，不是模型独占显存，不能直接用于推断参数存储成本。

## 自行实现的 DDPM

`src/models.py` 实现前向加噪、时间条件 U-Net、DDPM 后验更新与完整反向采样。训练目标是预测加入的标准高斯噪声，使用 MSE。

- 使用同一训练子集 50,000 张，不使用验证/测试图训练生成器；像素映射到 [-1,1]。
- 输入 1 通道、基础 32 通道，低分辨率层 64 通道；正弦时间编码、GroupNorm、SiLU、跳跃连接。
- 200 个扩散步，线性 beta 从 0.0001 到 0.05。终态仍非严格标准正态，报告列出残余信号。
- Adam，学习率 0.001，batch 256，20 epoch，训练种子 42，使用末轮权重，无按观感挑选。
- 预测 x0 截断到 [-1,1] 后计算 DDPM 后验均值；使用后验方差加噪，最后一步不加噪。

## 生成评价规则

固定初始噪声种子 123、反向随机数种子 124，完整展示 100 张样本，保存初始噪声和生成图张量。采样计时包括完整 200 步更新和过程快照，不包括后续绘图及最近邻搜索。

1. `samples.png`：全部样本，保持原始顺序，不筛选。
2. `forward.png`：前 8 张训练图在多个噪声级别的状态，同一列共用 epsilon。
3. `reverse.png`：同一批前 8 张生成图的完整采样过程快照。
4. `generation_metrics.png`：原图 CNN 对生成图的预测类别覆盖及训练损失。预测类别只是代理指标，不是真实标签或可靠质量分数。
5. `nearest.png`：前 10 张生成图与全部 50,000 张训练图中的像素 MSE 最近邻，交替展示；全部 100 张邻居索引和距离存于 JSON。
6. `failures.png`：按预定规则选择 CNN 最大概率最低的 20 张，标注原始编号及置信度。这是失败候选，不是人工失败标注；完整网格仍需检查高置信度伪影。

## 文件与复查

- `src/data.py`：沿用 Assignment3 的 IDX 读取及固定划分。
- `src/models.py`：MLP、CNN、时间去噪器、扩散过程。
- `run.py`：训练、评价、图表与权重保存。
- `verify.py`：置换等价性、形状、梯度、加噪公式、DDPM 后验和采样复现检查。
- `config/config.json`：所有主要超参数与种子。
- `outputs/classification.json` / `.csv`：分类准确率、训练代价、混淆矩阵、逐类召回率和错误类型。
- `outputs/*_architecture.json` / `*_history.json`：各层尺寸、参数与逐轮日志。
- `outputs/*_errors.png`：每组按测试顺序取前 20 个误判，展示原始图像。
- `outputs/diffusion_training.json` / `generation.json`：样本数、通道数、步数、轮数、设备、时间、最近邻和失败候选索引。
- `outputs/*.pt`：四组分类 checkpoint、扩散 checkpoint、初始噪声、生成样本，仅保留在本地，Git 忽略；运行 `python run.py` 可重新生成。
- `outputs/run.log`：完整控制台训练日志。
- `report/Assignment4_report.md`：图文报告，可用 `make_report.py` 从实验输出重新生成。

Git 提交保留源代码、配置和实验结果（JSON、CSV、PNG，以及复现置换所需的 `permutation.npy`）。权重与张量文件（PT/PTH/CKPT）、控制台日志（LOG）、缓存、下载数据、报告目录和 `task.md` 均不提交。逐轮训练记录仍以 `*_history.json` 保留，便于查看实际运行结果。

参考：Fashion-MNIST 官方数据；Ho et al., *Denoising Diffusion Probabilistic Models* (2020)。本实验未使用预训练生成器，也未计算未经验证的 FID。

