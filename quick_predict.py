#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速预测脚本 - 使用A-YOLOM预训练模型进行推理
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("A-YOLOM 快速预测")
    print("=" * 60 + "\n")

    # 配置参数
    # ========== 请修改以下路径 ==========
    MODEL_PATH = "models/best.pt"  # 预训练模型路径
    SOURCE = "path/to/your/images"  # 输入图片或视频路径
    # ===================================

    # 高级参数（可选修改）
    IMGSZ = (384, 672)      # 图像尺寸 (官方推荐)
    DEVICE = 0              # GPU设备 (0, 1, 2...) 或 'cpu'
    CONF = 0.25             # 置信度阈值
    IOU = 0.45              # NMS IOU阈值
    SAVE = True             # 保存结果
    SHOW_LABELS = False     # 显示标签

    # 检查模型文件
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"✗ 错误: 找不到模型文件 '{MODEL_PATH}'")
        print("\n请执行以下步骤:")
        print("1. 从以下链接下载预训练模型:")
        print("   https://uwin365-my.sharepoint.com/:f:/g/personal/wang621_uwindsor_ca/EoUsIoFgcEhBgnO4kTjDQG4BUUSHMFXG4ami9qjvTUTofA?e=0xuAJG")
        print(f"2. 将下载的 'best.pt' 放到 '{MODEL_PATH}' 路径")
        print(f"3. 或修改脚本中的 MODEL_PATH 变量\n")
        return

    # 检查输入路径
    source_path = Path(SOURCE)
    if not source_path.exists():
        print(f"✗ 错误: 找不到输入路径 '{SOURCE}'")
        print("\n请修改脚本中的 SOURCE 变量，指向你的图片或视频路径\n")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    try:
        model = YOLO(str(model_path))
        print("✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}\n")
        return

    # 运行预测
    print("开始预测...")
    print(f"  输入: {SOURCE}")
    print(f"  图像尺寸: {IMGSZ}")
    print(f"  设备: {DEVICE}")
    print(f"  置信度阈值: {CONF}")
    print(f"  IOU阈值: {IOU}\n")

    try:
        results = model.predict(
            source=str(source_path),
            imgsz=IMGSZ,
            device=DEVICE,
            save=SAVE,
            conf=CONF,
            iou=IOU,
            show_labels=SHOW_LABELS
        )

        print("=" * 60)
        print("✓ 预测完成！")
        print(f"  处理图片数: {len(results)}")

        if SAVE:
            # 查找保存路径
            from ultralytics.yolo.cfg import get_save_dir
            print(f"  结果保存在: runs/detect/predict* 目录下")

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"✗ 预测失败: {e}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
