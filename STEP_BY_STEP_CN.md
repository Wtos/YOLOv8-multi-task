# YOLOv8 Multi-Task 部署教程 - 一步一步指导

本教程将带你从零开始部署A-YOLOM项目，每一步都有详细说明。

---

## 📋 准备工作

### 检查前置条件

1. **检查是否安装了Conda**
   ```bash
   conda --version
   ```

   **预期输出类似：**
   ```
   conda 23.x.x
   ```

   **如果提示"找不到命令"：**
   - 需要先安装 [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - 安装后重启终端

2. **检查GPU（如果有的话）**
   ```bash
   nvidia-smi
   ```

   **如果有GPU，会看到类似：**
   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        ...          |
   ```

   **记住你的CUDA版本**（上面例子是12.0）

   **如果没有GPU或命令报错：** 不影响使用，后续使用CPU模式即可

---

## 📥 第1步：获取项目代码

### 方式A：如果你已经在项目目录中（推荐）

```bash
# 确认你在正确的目录
pwd
```

**应该显示类似：**
```
/home/user/YOLOv8-multi-task
```

**检查文件是否存在：**
```bash
ls -la
```

**应该看到：**
```
environment.yml
QUICK_START_CN.md
check_environment.py
quick_predict.py
ultralytics/
requirements.txt
setup.py
...
```

✅ **如果看到这些文件，说明你已经在正确的位置，继续下一步！**

### 方式B：如果需要重新克隆

```bash
# 克隆仓库
git clone https://github.com/Wtos/YOLOv8-multi-task.git

# 进入项目目录
cd YOLOv8-multi-task

# 切换到更新后的分支（包含我刚才的改进）
git checkout claude/check-deployment-setup-01VEy8LNM7xuxWUfBzmYMzz4
```

---

## 🐍 第2步：创建Conda环境

### 2.1 使用environment.yml创建环境（推荐）

```bash
# 确保你在项目根目录
cd /home/user/YOLOv8-multi-task

# 创建conda环境
conda env create -f environment.yml
```

**这个过程会：**
- 创建名为 `yolom` 的虚拟环境
- 安装Python 3.7.16
- 安装PyTorch 1.13.1和所有依赖包
- **需要5-15分钟**，取决于网络速度

**预期输出：**
```
Collecting package metadata...
Solving environment...
...
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use:
#
#     $ conda activate yolom
#
```

### 2.2 激活环境

```bash
conda activate yolom
```

**成功后，命令提示符会变成：**
```
(yolom) user@hostname:~/YOLOv8-multi-task$
```

注意前面多了 `(yolom)` 标记！

### 2.3 验证Python版本

```bash
python --version
```

**应该显示：**
```
Python 3.7.16
```

---

## 📦 第3步：安装项目

### 3.1 安装PyTorch（根据你的GPU情况）

**如果你有GPU且CUDA版本是11.7或11.8：**
```bash
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

**如果你只有CPU或想先用CPU测试：**
```bash
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
```

**如果你的CUDA是其他版本：**
- CUDA 11.6: 把 `cu117` 改成 `cu116`
- CUDA 11.3: 把 `cu117` 改成 `cu113`

**等待安装完成**（可能需要几分钟）

### 3.2 安装项目依赖

```bash
# 确保在项目根目录
cd /home/user/YOLOv8-multi-task

# 以开发模式安装项目
pip install -e .
```

**预期输出：**
```
Obtaining file:///home/user/YOLOv8-multi-task
...
Successfully installed ultralytics-x.x.x ...
```

---

## ✅ 第4步：验证环境

运行环境检查脚本：

```bash
python check_environment.py
```

**预期输出（如果一切正常）：**
```
============================================================
YOLOv8 Multi-Task 环境检查
============================================================

============================================================
检查Python版本...
✓ Python版本: 3.7.16
✓ Python版本符合要求 (3.7.x)
============================================================
检查PyTorch...
✓ PyTorch版本: 1.13.1+cu117
✓ CUDA可用
  - CUDA版本: 11.7
  - GPU设备数: 1
  - GPU 0: NVIDIA GeForce RTX 3090
============================================================
检查依赖包...
✓ matplotlib>=3.2.2
✓ opencv-python>=4.6.0
✓ Pillow>=7.1.2
...
============================================================
检查项目结构...
✓ ultralytics
✓ ultralytics/models
✓ ultralytics/datasets
✓ requirements.txt
✓ setup.py
============================================================
检查Ultralytics YOLO模块...
✓ Ultralytics YOLO模块导入成功
============================================================
检查结果汇总
============================================================
Python版本     : ✓ 通过
PyTorch        : ✓ 通过
依赖包         : ✓ 通过
项目结构       : ✓ 通过
YOLO模块       : ✓ 通过
============================================================
✓ 所有检查通过！环境配置正确。
```

**如果有错误：**
- 仔细阅读错误提示
- 根据提示执行建议的修复命令
- 重新运行 `python check_environment.py`

---

## 📥 第5步：下载预训练模型

### 5.1 创建模型目录

```bash
mkdir -p models
```

### 5.2 下载模型文件

1. **打开浏览器**，访问：
   ```
   https://uwin365-my.sharepoint.com/:f:/g/personal/wang621_uwindsor_ca/EoUsIoFgcEhBgnO4kTjDQG4BUUSHMFXG4ami9qjvTUTofA?e=0xuAJG
   ```

2. **选择模型版本：**
   - `A-YOLOM(n)` - 轻量级版本（4.43M参数，速度快）← 推荐先下载这个
   - `A-YOLOM(s)` - 标准版本（13.61M参数，精度更高）

3. **下载best.pt文件**

4. **将下载的文件移动到项目目录：**
   ```bash
   # 假设你下载到了 ~/Downloads/best.pt
   mv ~/Downloads/best.pt /home/user/YOLOv8-multi-task/models/best.pt
   ```

5. **验证文件存在：**
   ```bash
   ls -lh models/best.pt
   ```

   **应该显示：**
   ```
   -rw-r--r-- 1 user user 8.9M Nov 26 10:00 models/best.pt
   ```

---

## 🚀 第6步：运行预测（测试模型）

### 6.1 准备测试图片

**选项A：使用自己的图片**

```bash
# 创建测试图片目录
mkdir -p test_images

# 复制你的图片到这个目录
cp /path/to/your/images/*.jpg test_images/
```

**选项B：从网上下载测试图片**

```bash
# 创建测试目录
mkdir -p test_images

# 下载一张道路测试图片（示例）
wget -O test_images/road_test.jpg "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1280"
```

### 6.2 修改预测脚本

```bash
# 用文本编辑器打开脚本
nano quick_predict.py
# 或使用其他编辑器：vim, gedit, code, etc.
```

**找到这两行并修改：**
```python
MODEL_PATH = "models/best.pt"           # 如果模型在这个位置，不用改
SOURCE = "test_images"                  # 改成你的图片目录路径
```

**如果使用CPU，还需要修改：**
```python
DEVICE = 'cpu'                          # 将 0 改成 'cpu'
```

**保存并退出：**
- nano: 按 `Ctrl+X`，然后 `Y`，然后 `Enter`
- vim: 按 `Esc`，输入 `:wq`，然后 `Enter`

### 6.3 运行预测

```bash
python quick_predict.py
```

**预期输出：**
```
============================================================
A-YOLOM 快速预测
============================================================

加载模型: models/best.pt
✓ 模型加载成功

开始预测...
  输入: test_images
  图像尺寸: (384, 672)
  设备: 0
  置信度阈值: 0.25
  IOU阈值: 0.45

image 1/5 /home/user/YOLOv8-multi-task/test_images/road_test.jpg: 384x672 1 vehicle, 2 lanes, 1 drivable area, 45.2ms
...
============================================================
✓ 预测完成！
  处理图片数: 5
  结果保存在: runs/detect/predict* 目录下
============================================================
```

### 6.4 查看结果

```bash
# 查看生成的预测结果
ls -la runs/detect/predict*/
```

**结果图片会保存在：**
```
runs/detect/predict/
runs/detect/predict2/
...
```

**使用图片查看器打开结果：**
```bash
# Linux
xdg-open runs/detect/predict/road_test.jpg

# 或直接用文件管理器浏览该目录
```

---

## 🎯 第7步：使用项目自带脚本（可选）

### 7.1 使用predict.py预测

```bash
# 编辑脚本
nano ultralytics/predict.py
```

**修改这些行：**
```python
# 第15行附近 - 使用下载的预训练模型
model = YOLO('models/best.pt')

# 第24行 - 设置你的图片路径
source='test_images',  # 你的图片目录

# 第26行 - 如果用CPU
device='cpu',  # 或保持 [0] 使用GPU
```

**运行：**
```bash
python ultralytics/predict.py
```

### 7.2 训练模型（需要数据集）

⚠️ **注意：** 训练需要先下载BDD100K数据集（约10GB+）

**如果你有数据集：**

1. **下载数据集**（参考README.md的链接）

2. **组织数据集目录结构：**
   ```
   dataset_root/
   ├── images/
   │   ├── train2017/
   │   └── val2017/
   ├── detection-object/
   ├── seg-drivable-10/
   └── seg-lane-11/
   ```

3. **修改数据集配置：**
   ```bash
   nano ultralytics/datasets/bdd-multi.yaml
   ```

   更新 `path` 字段为你的数据集路径

4. **运行训练：**
   ```bash
   python ultralytics/train.py
   ```

---

## 🔧 常见问题解决

### 问题1：ModuleNotFoundError: No module named 'ultralytics'

**解决：**
```bash
cd /home/user/YOLOv8-multi-task
pip install -e .
```

### 问题2：CUDA out of memory

**解决：**
- 使用CPU模式：将 `device=0` 改为 `device='cpu'`
- 或减小batch size（如果在训练）

### 问题3：找不到模型文件

**解决：**
```bash
# 检查模型文件是否存在
ls -la models/best.pt

# 如果不存在，重新下载并放到正确位置
```

### 问题4：conda activate yolom 无效

**解决：**
```bash
# 初始化conda
conda init bash
# 重启终端
source ~/.bashrc
# 再次激活
conda activate yolom
```

---

## 📊 成功标志

如果你完成了以上步骤，你应该：

✅ 成功创建了 `yolom` conda环境
✅ 所有依赖包安装完成
✅ 环境检查脚本全部通过
✅ 下载了预训练模型
✅ 成功运行预测并生成结果图片

---

## 🎉 下一步

现在你已经成功部署了项目！你可以：

1. **测试更多图片** - 将你自己的道路图片放到 `test_images` 目录测试
2. **调整参数** - 修改置信度阈值、IOU阈值等参数
3. **准备数据集** - 如果要训练，下载BDD100K数据集
4. **尝试Docker部署** - 告诉我，我帮你配置Docker版本

---

## 📞 需要帮助？

如果某一步遇到问题：

1. 仔细阅读错误信息
2. 检查是否激活了conda环境 `(yolom)`
3. 确认在正确的目录 `/home/user/YOLOv8-multi-task`
4. 告诉我具体在哪一步遇到了什么错误，我会帮你解决

---

**祝你使用顺利！🚀**
