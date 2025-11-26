#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检查脚本 - 验证YOLOv8 Multi-Task项目环境是否正确配置
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查Python版本...")
    version = sys.version_info
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor == 7:
        print("✓ Python版本符合要求 (3.7.x)")
        return True
    else:
        print(f"⚠ 警告: 推荐使用Python 3.7.16，当前版本可能存在兼容性问题")
        return False

def check_pytorch():
    """检查PyTorch安装"""
    print("=" * 60)
    print("检查PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")

        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU设备数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA不可用，将使用CPU模式")

        return True
    except ImportError:
        print("✗ PyTorch未安装")
        print("  请运行: pip install torch==1.13.1 torchvision==0.14.1")
        return False

def check_dependencies():
    """检查其他依赖包"""
    print("=" * 60)
    print("检查依赖包...")

    required_packages = {
        'matplotlib': 'matplotlib>=3.2.2',
        'cv2': 'opencv-python>=4.6.0',
        'PIL': 'Pillow>=7.1.2',
        'yaml': 'PyYAML>=5.3.1',
        'requests': 'requests>=2.23.0',
        'scipy': 'scipy>=1.4.1',
        'tqdm': 'tqdm>=4.64.0',
        'pandas': 'pandas>=1.1.4',
        'seaborn': 'seaborn>=0.11.0',
    }

    all_installed = True
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'cv2':
                import cv2
            elif module_name == 'PIL':
                import PIL
            elif module_name == 'yaml':
                import yaml
            else:
                __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} 未安装")
            all_installed = False

    return all_installed

def check_ultralytics():
    """检查Ultralytics YOLO模块"""
    print("=" * 60)
    print("检查Ultralytics YOLO模块...")
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ Ultralytics YOLO模块导入失败: {e}")
        print("  请在项目根目录运行: pip install -e .")
        return False

def check_project_structure():
    """检查项目结构"""
    print("=" * 60)
    print("检查项目结构...")

    project_root = Path(__file__).resolve().parent
    required_paths = [
        'ultralytics',
        'ultralytics/models',
        'ultralytics/datasets',
        'requirements.txt',
        'setup.py',
    ]

    all_exists = True
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} 不存在")
            all_exists = False

    return all_exists

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("YOLOv8 Multi-Task 环境检查")
    print("=" * 60 + "\n")

    results = []

    # 执行检查
    results.append(("Python版本", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("依赖包", check_dependencies()))
    results.append(("项目结构", check_project_structure()))
    results.append(("YOLO模块", check_ultralytics()))

    # 总结
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)

    for name, status in results:
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{name:15s}: {status_str}")

    all_passed = all(status for _, status in results)

    print("=" * 60)
    if all_passed:
        print("✓ 所有检查通过！环境配置正确。")
        print("\n下一步:")
        print("1. 下载预训练模型")
        print("2. 运行 quick_predict.py 进行推理测试")
    else:
        print("⚠ 部分检查未通过，请根据提示修复问题。")
        print("\n建议操作:")
        print("1. 确保在项目根目录")
        print("2. 运行: pip install -e .")
        print("3. 重新运行本脚本")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
