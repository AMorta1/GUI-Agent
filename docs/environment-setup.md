# GUI Agent 开发环境配置文档

## 1. 配置目标

本环境用于“基于多模态大模型的桌面 GUI 智能体开发与优化”项目，当前覆盖：

- Python 3.10 独立开发环境；
- PyTorch 2.2+ 与 NVIDIA CUDA 加速；
- 第 2 周所需的跨平台截图、桌面控制、OpenCV、PyQt5 和 EasyOCR；
- 可重复执行的 PyTorch、CUDA、基础库和内存截图验证。

第 2 周 OCR 后端已确定为 EasyOCR。Agent 框架、模型部署和 LoRA 相关依赖将在对应开发阶段根据实际选型加入，避免提前引入不必要的依赖冲突。

## 2. 当前验证环境

验证日期：2026-09-20

| 项目 | 当前配置 |
|---|---|
| 操作系统 | Windows 11，AMD64 |
| Conda | 25.5.1 |
| Conda 环境 | `gui-agent` |
| Python | 3.10.21 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU 显存 | 8188 MiB |
| Compute Capability | 8.9 |
| NVIDIA 驱动 | 581.29 |
| `nvidia-smi` 最高 CUDA 版本 | 13.0 |
| PyTorch | 2.13.0+cu130 |
| TorchVision | 0.28.0+cu130 |
| PyTorch CUDA Runtime | 13.0 |
| 下载模型缓存根目录 | `E:\Projects\Microsoft\.model-cache` |

`nvidia-smi` 中的 CUDA 版本表示驱动支持的最高 CUDA 版本，并不表示系统已经安装同版本 CUDA Toolkit。本机当前没有检测到 `nvcc`，但 PyTorch 官方二进制包已携带所需 CUDA Runtime，因此不影响当前模型推理和训练。只有后续需要编译自定义 CUDA 扩展时，才需要单独安装 CUDA Toolkit 和相应编译工具。

## 3. 创建环境

在 Anaconda Prompt 或已经初始化 Conda 的 PowerShell 中运行：

```powershell
cd E:\Projects\Microsoft\GUI-Agent
conda env create -f environment.yml
conda activate gui-agent
python -m pip install -r requirements/torch-cu130.txt
python -m pip install -r requirements/base.txt
```

如果 `gui-agent` 环境已经存在，则使用：

```powershell
conda activate gui-agent
python -m pip install -r requirements/torch-cu130.txt
python -m pip install -r requirements/base.txt
```

CUDA 13.0 的 PyTorch 安装文件适用于当前 Windows + NVIDIA 环境。其他计算平台应根据 PyTorch 官方安装页面选择对应构建，不能直接照搬 CUDA 13.0 配置。

## 4. 自动验证

激活环境后运行：

```powershell
python scripts/verify_environment.py --require-cuda
```

脚本会执行以下检查：

1. Python 版本和解释器路径；
2. 基础工具库导入和版本；
3. PyAutoGUI 读取屏幕尺寸；
4. mss 在内存中截取一帧并由 OpenCV 转换；
5. PyTorch CUDA 可用性和 GPU 信息；
6. 在 GPU 上执行矩阵乘法并检查结果是否为有限值。
7. EasyOCR 安装版本和导入状态。

脚本不会保存屏幕截图，也不会移动鼠标或发送键盘输入。无图形界面的环境可以使用 `--skip-screen` 跳过截图测试。

## 5. 本机验证结果

当前机器验证通过：

| 验证项 | 结果 |
|---|---|
| `torch.cuda.is_available()` | `True` |
| CUDA 设备数量 | 1 |
| GPU 名称 | NVIDIA GeForce RTX 4070 Laptop GPU |
| PyTorch CUDA Runtime | 13.0 |
| GPU 矩阵乘法 | 通过，结果均为有限值 |
| 基础库导入 | 通过 |
| PyAutoGUI 屏幕尺寸 | 2560 × 1600 |
| Windows 显示缩放 | 150% |
| mss 内存截图 | 通过，输出数组为 1600 × 2560 × 3 |
| EasyOCR | 1.7.2，CUDA 模型初始化和英文合成图推理通过 |
| `pip check` | 通过，无依赖冲突 |

## 6. 后续阶段依赖

以下依赖属于项目大纲，但本阶段暂不锁定版本：

| 阶段 | 待加入依赖 |
|---|---|
| 第 2 周 OCR | EasyOCR 1.7.2（已加入）；PaddleOCR 仅作为有余力时的对比项 |
| 第 3 周 Agent 与数据 | LangChain/LlamaIndex、Transformers、Datasets、Pandas |
| 第 5 周模型微调 | PEFT、Accelerate、bitsandbytes |
| 第 7 周评估 | Matplotlib、Seaborn |

每组依赖会在对应模块开始前，根据模型、GPU 和跨平台兼容性确定版本并单独记录。

## 7. 下载模型缓存

本机将下载的第三方基础模型统一放在：

```text
E:\Projects\Microsoft\.model-cache
```

EasyOCR 使用用户级环境变量：

```text
EASYOCR_MODULE_PATH=E:\Projects\Microsoft\.model-cache\easyocr
```

其模型文件实际位于 `E:\Projects\Microsoft\.model-cache\easyocr\model`。当前模型已从 EasyOCR 默认的用户缓存迁移到该目录，并通过 CUDA 推理复测。

该绝对路径只属于本机环境配置，不写入业务代码。其他计算机可以继续使用各工具的默认缓存路径，也可以设置适合本机的路径。后续引入 Hugging Face、Torch Hub 等工具时，分别配置其官方缓存环境变量和独立子目录。

下载的基础模型缓存不提交 Git。项目训练产生的 LoRA 权重、检查点和实验结果属于项目产物，不放入 `.model-cache`，其目录在对应开发阶段另行确定。

## 8. 常见问题

### `nvidia-smi` 显示 CUDA，但 `nvcc` 不存在

这是正常情况。NVIDIA 驱动和 CUDA Toolkit 是两个不同组件。预编译 PyTorch 使用包内 CUDA Runtime，不要求系统存在 `nvcc`。

### PyTorch 无法识别 GPU

依次检查：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

确认当前终端已经激活 `gui-agent`，并且安装的是 CUDA 构建而不是 CPU 构建。

### 8GB 显存的限制

8GB 显存满足项目大纲的推荐配置，但不代表能够以全精度直接部署或微调所有 9B/11B 多模态模型。后续模型选择需要结合量化、LoRA/QLoRA、梯度检查点或云端 GPU 资源进行实际验证。

## 9. 官方参考

- [PyTorch - Start Locally](https://pytorch.org/get-started/locally/)
- [PyTorch - Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
