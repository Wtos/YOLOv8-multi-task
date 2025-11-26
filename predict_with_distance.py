#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
带距离估算的预测脚本
在检测结果上显示距离信息
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from distance_estimator import DistanceEstimator


def process_video_with_distance(
    model_path,
    video_path,
    output_path=None,
    focal_length=1000,
    show_pixel_distance=True,
    show_real_distance=True,
    conf_threshold=0.25
):
    """
    处理视频并添加距离信息

    Args:
        model_path: 模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        focal_length: 相机焦距
        show_pixel_distance: 是否显示像素距离
        show_real_distance: 是否显示真实距离
        conf_threshold: 置信度阈值
    """

    # 加载模型
    print(f"\n{'='*60}")
    print("A-YOLOM 带距离估算的预测")
    print(f"{'='*60}\n")
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    print("✓ 模型加载成功\n")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  焦距: {focal_length}px\n")

    # 创建距离估算器
    estimator = DistanceEstimator(
        focal_length=focal_length,
        image_width=width,
        image_height=height
    )

    # 设置输出视频
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"runs/predict_distance/{video_name}_with_distance.mp4"

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("开始处理...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 运行检测
        results = model(frame, conf=conf_threshold, verbose=False)

        # 处理分割结果（车道线和可行驶区域）
        # 多任务模型返回: [检测结果, 分割mask1, 分割mask2]
        if isinstance(results, list) and len(results) > 0:
            # 先处理分割结果作为底层
            if len(results) >= 3:
                try:
                    # 获取分割masks
                    mask1 = results[1][0] if isinstance(results[1], list) else results[1]
                    mask2 = results[2][0] if isinstance(results[2], list) else results[2]

                    # 转换为numpy数组
                    if torch.is_tensor(mask1):
                        mask1_np = mask1.cpu().numpy().astype(np.uint8)
                    else:
                        mask1_np = np.array(mask1).astype(np.uint8)

                    if torch.is_tensor(mask2):
                        mask2_np = mask2.cpu().numpy().astype(np.uint8)
                    else:
                        mask2_np = np.array(mask2).astype(np.uint8)

                    # Resize masks to match frame size if needed
                    if mask1_np.shape[:2] != (height, width):
                        mask1_np = cv2.resize(mask1_np, (width, height), interpolation=cv2.INTER_NEAREST)
                    if mask2_np.shape[:2] != (height, width):
                        mask2_np = cv2.resize(mask2_np, (width, height), interpolation=cv2.INTER_NEAREST)

                    # 创建彩色mask overlay
                    # mask1: 绿色 (可行驶区域)
                    # mask2: 红色 (车道线)
                    color_mask1 = np.zeros_like(frame)
                    color_mask1[mask1_np > 0] = [0, 255, 0]  # 绿色

                    color_mask2 = np.zeros_like(frame)
                    color_mask2[mask2_np > 0] = [255, 0, 0]  # 红色

                    # 叠加到原图上（半透明）
                    alpha = 0.3
                    frame = cv2.addWeighted(frame, 1, color_mask1, alpha, 0)
                    frame = cv2.addWeighted(frame, 1, color_mask2, alpha, 0)
                except Exception as e:
                    # 如果分割处理失败，继续处理检测
                    pass

            # 处理检测结果
            det_result = results[0] if not isinstance(results[0], list) else results[0][0]
            boxes = det_result.boxes if hasattr(det_result, 'boxes') else None
        else:
            boxes = None

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 获取检测框信息
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # 获取类别名
                # 对于single_cls=True的模型，所有物体可能都被标记为同一类
                # 根据bbox大小启发式判断车辆类型
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height

                if hasattr(model, 'names') and model.names:
                    base_name = model.names[cls]
                else:
                    base_name = 'object'

                # 启发式判断车型（基于bbox大小）- 针对道路场景优化
                # 主要区分car和truck
                if bbox_area > 30000:  # 大型车辆 - 调整阈值
                    class_name = 'truck'
                elif bbox_area > 8000:  # 普通轿车 - 降低阈值以匹配更多car
                    class_name = 'car'
                elif bbox_area > 3000:  # 小型目标
                    class_name = 'person' if bbox_height > bbox_width else 'bicycle'
                else:
                    class_name = base_name

                # 估算距离
                bbox = [x1, y1, x2, y2]
                dist_info = estimator.get_distance_info(bbox, class_name)

                # 绘制检测框
                color = (0, 255, 255)  # 黄色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 准备标签文本
                labels = []
                labels.append(f"{class_name} {conf:.2f}")

                if show_real_distance:
                    labels.append(f"Dist: {dist_info['real_distance']:.1f}m")

                if show_pixel_distance:
                    labels.append(f"({dist_info['pixel_distance']:.0f}px)")

                # 绘制标签
                label_text = " | ".join(labels)

                # 计算文本大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )

                # 绘制文本背景
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - baseline - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # 黑色文字
                    thickness
                )

                # 标记车辆底部中点（接地位置）- 用于距离参考
                bottom_center_x = (x1 + x2) // 2
                bottom_center_y = y2  # 检测框底部
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 4, (0, 255, 0), -1)

        # 写入帧
        out.write(frame)

        # 显示进度
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")

    # 清理
    cap.release()
    out.release()

    print(f"\n{'='*60}")
    print("✓ 处理完成！")
    print(f"  输出文件: {output_path}")
    print(f"{'='*60}\n")


def main():
    """主函数"""

    # ========== 配置参数 ==========
    # 模型路径
    MODEL_PATH = "/home/chengpeng/project/cuda_acceleration/chapter7-deploy-yolo-detection/YOLOv8-multi-task/models/best.pt"

    # 输入视频路径
    VIDEO_PATH = "/home/chengpeng/project/cuda_acceleration/chapter7-deploy-yolo-detection/ultralytics/ultralytics/assets/car_8.mp4"

    # 输出路径（可选，默认会自动生成）
    OUTPUT_PATH = None  # 或指定如 "output_video.mp4"

    # 距离估算参数
    FOCAL_LENGTH = 1000  # 焦距（像素）- 建议通过校准获得更准确的值

    # 显示选项
    SHOW_PIXEL_DISTANCE = False  # 是否显示像素距离
    SHOW_REAL_DISTANCE = True   # 是否显示真实距离

    # 检测参数
    CONF_THRESHOLD = 0.25  # 置信度阈值
    # =============================

    # 检查文件是否存在
    if not Path(MODEL_PATH).exists():
        print(f"✗ 错误: 找不到模型文件 '{MODEL_PATH}'")
        return

    if not Path(VIDEO_PATH).exists():
        print(f"✗ 错误: 找不到视频文件 '{VIDEO_PATH}'")
        return

    # 处理视频
    process_video_with_distance(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        focal_length=FOCAL_LENGTH,
        show_pixel_distance=SHOW_PIXEL_DISTANCE,
        show_real_distance=SHOW_REAL_DISTANCE,
        conf_threshold=CONF_THRESHOLD
    )


if __name__ == "__main__":
    main()
