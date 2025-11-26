# YOLOv8 Multi-Task 快速部署指南

本指南帮助你快速在本地部署 A-YOLOM 项目并运行预训练模型。

## 📋 前置要求

- Anaconda 或 Miniconda
- NVIDIA GPU（推荐，也支持CPU）
- CUDA 11.7（如果使用GPU）

## 🚀 快速开始（Conda方式）

### 1. 创建并激活虚拟环境

#### 方式A：使用 environment.yml（推荐）

```bash
# 克隆或进入项目目录
cd YOLOv8-multi-task

# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate yolom
```

#### 方式B：手动创建环境

```bash
# 创建Python 3.7.16环境
conda create -n yolom python=3.7.16
conda activate yolom

# 安装PyTorch（GPU版本 - CUDA 11.7）
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# 或者安装CPU版本
# pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
```

### 2. 安装项目依赖

```bash
# 进入项目根目录
cd YOLOv8-multi-task

# 以开发模式安装
pip install -e .
```

### 3. 下载预训练模型

从以下链接下载预训练模型（选择n或s版本）：
- [A-YOLOM 预训练模型](https://uwin365-my.sharepoint.com/:f:/g/personal/wang621_uwindsor_ca/EoUsIoFgcEhBgnO4kTjDQG4BUUSHMFXG4ami9qjvTUTofA?e=0xuAJG)

下载后将 `best.pt` 放到项目目录中，例如：
```
YOLOv8-multi-task/
├── models/
│   └── best.pt  # 下载的预训练模型
```

### 4. 运行预测

创建一个简单的预测脚本 `quick_predict.py`：

```python
from pathlib import Path
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/best.pt')

# 预测单张图片或文件夹
results = model.predict(
    source='path/to/your/images',  # 改为你的图片路径
    imgsz=(384, 672),     # 官方推荐尺寸
    device=0,             # GPU设备ID，CPU使用'cpu'
    save=True,            # 保存结果
    conf=0.25,            # 置信度阈值
    iou=0.45,             # NMS IOU阈值
    show_labels=False     # 不显示标签
)

print("预测完成！结果保存在 runs/detect/predict 目录下")
```

运行预测：
```bash
python quick_predict.py
```

## 📊 使用项目自带脚本

### 训练模型

如果你有BDD100K数据集，可以使用训练脚本：

```bash
# 1. 准备数据集（参考README.md的数据集结构）
# 2. 更新数据集路径：ultralytics/datasets/bdd-multi.yaml
# 3. 运行训练
python ultralytics/train.py
```

**注意**：需要在 `ultralytics/train.py` 中调整以下参数：
- `device=[0]` - 你的GPU设备ID
- `batch=12` - 根据GPU显存调整批次大小

### 验证模型

```bash
python ultralytics/val.py
```

### 预测/推理

```bash
# 修改 ultralytics/predict.py 中的配置：
# - model_path: 预训练模型路径
# - source: 输入图片/视频路径
python ultralytics/predict.py
```

## ⚙️ 配置说明

### GPU设置

```python
device=[0]        # 使用单个GPU (GPU 0)
device=[0,1,2]    # 使用多个GPU
device='cpu'      # 使用CPU
```

### 图像尺寸

对于官方预训练模型，推荐使用：
- 输入图片尺寸：(720, 1280)
- 模型处理尺寸：`imgsz=(384, 672)`

## 🔍 验证安装

运行以下Python代码验证安装：

```python
import torch
from ultralytics import YOLO

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 测试YOLO导入
print("YOLO模块导入成功！")
```

## 🐛 常见问题

### 1. 显存不足 (CUDA out of memory)

解决方法：
- 减小 `batch` 参数
- 减小 `imgsz` 尺寸
- 使用更小的模型（n版本而非s版本）

### 2. 找不到模块错误

```bash
# 确保在项目根目录执行了
pip install -e .
```

### 3. CUDA版本不匹配

检查你的CUDA版本：
```bash
nvidia-smi
```

根据CUDA版本安装对应的PyTorch：
- CUDA 11.7: `--index-url https://download.pytorch.org/whl/cu117`
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CPU only: `--index-url https://download.pytorch.org/whl/cpu`

## 📝 下一步

1. ✅ Conda环境部署完成
2. 🔄 准备尝试Docker部署（见 `DOCKER_DEPLOYMENT.md`）
3. 📚 详细使用说明参考 `README.md`

## 💡 提示

- 如果只是测试预训练模型，不需要下载BDD100K数据集
- 建议先用小批量图片测试，确认环境正常后再大规模使用
- 训练模型需要下载完整的BDD100K数据集（约10GB+）

## 📧 问题反馈

如遇到问题，请检查：
1. Python版本是否为3.7.16
2. PyTorch版本是否为1.13.1
3. CUDA版本是否兼容
4. 是否在项目根目录执行命令
