#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
距离估算器 - 基于检测框尺寸估算物体距离
改进版：针对不同类别使用不同的真实尺寸
"""

import math


class DistanceEstimator:
    def __init__(self, focal_length=1000, image_width=1280, image_height=720):
        """
        初始化距离估算器

        Args:
            focal_length: 相机焦距（像素），建议通过校准获得
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
        """
        self.focal_length = focal_length
        self.image_width = image_width
        self.image_height = image_height

        # 不同类别物体的真实尺寸（米）
        # 格式：'类别名': {'width': 宽度, 'height': 高度}
        self.object_dimensions = {
            # 车辆类
            'car': {'width': 1.8, 'height': 1.5},
            'sedan': {'width': 1.8, 'height': 1.4},
            'suv': {'width': 2.0, 'height': 1.7},
            'truck': {'width': 2.5, 'height': 3.0},
            'bus': {'width': 2.6, 'height': 3.2},
            'van': {'width': 2.0, 'height': 2.0},

            # 两轮车
            'bicycle': {'width': 0.6, 'height': 1.2},
            'motorcycle': {'width': 0.8, 'height': 1.3},

            # 行人
            'person': {'width': 0.5, 'height': 1.7},

            # 交通设施
            'traffic light': {'width': 0.3, 'height': 0.8},
            'stop sign': {'width': 0.75, 'height': 0.75},

            # 默认（未知类别）
            'unknown': {'width': 2.0, 'height': 1.5},
        }

    def estimate_distance_by_width(self, bbox_width, object_class='car'):
        """
        基于检测框宽度估算距离

        Args:
            bbox_width: 检测框宽度（像素）
            object_class: 物体类别名称

        Returns:
            distance: 估算距离（米）
        """
        # 获取该类别的真实宽度，如果不存在则使用默认值
        real_width = self.object_dimensions.get(
            object_class,
            self.object_dimensions['unknown']
        )['width']

        # 距离 = (真实宽度 × 焦距) / 像素宽度
        if bbox_width > 0:
            distance = (real_width * self.focal_length) / bbox_width
        else:
            distance = 100  # 默认远距离

        return distance

    def estimate_distance_by_height(self, bbox_height, object_class='car'):
        """
        基于检测框高度估算距离（通常比宽度更稳定）

        Args:
            bbox_height: 检测框高度（像素）
            object_class: 物体类别名称

        Returns:
            distance: 估算距离（米）
        """
        # 获取该类别的真实高度
        real_height = self.object_dimensions.get(
            object_class,
            self.object_dimensions['unknown']
        )['height']

        # 距离 = (真实高度 × 焦距) / 像素高度
        if bbox_height > 0:
            distance = (real_height * self.focal_length) / bbox_height
        else:
            distance = 100

        return distance

    def estimate_distance(self, bbox, object_class='car', method='hybrid'):
        """
        综合估算距离

        Args:
            bbox: [x1, y1, x2, y2] 检测框坐标
            object_class: 物体类别名称
            method: 估算方法
                - 'width': 只用宽度
                - 'height': 只用高度
                - 'hybrid': 混合方法（推荐）

        Returns:
            distance: 估算距离（米）
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if method == 'width':
            distance = self.estimate_distance_by_width(bbox_width, object_class)

        elif method == 'height':
            distance = self.estimate_distance_by_height(bbox_height, object_class)

        else:  # hybrid
            dist_width = self.estimate_distance_by_width(bbox_width, object_class)
            dist_height = self.estimate_distance_by_height(bbox_height, object_class)

            # 加权平均（高度通常更稳定，给更高权重）
            distance = 0.4 * dist_width + 0.6 * dist_height

        # 限制在合理范围（1-100米）
        return max(1.0, min(distance, 100.0))

    def get_pixel_distance(self, bbox):
        """
        计算检测框中心到图像中心的像素距离

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            pixel_distance: 像素距离
            center: (center_x, center_y) 检测框中心坐标
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 图像中心
        img_center_x = self.image_width / 2
        img_center_y = self.image_height / 2

        # 欧几里得距离
        pixel_distance = math.sqrt(
            (center_x - img_center_x)**2 +
            (center_y - img_center_y)**2
        )

        return pixel_distance, (center_x, center_y)

    def get_distance_info(self, bbox, object_class='car'):
        """
        获取完整的距离信息

        Args:
            bbox: [x1, y1, x2, y2]
            object_class: 物体类别

        Returns:
            dict: 包含各种距离信息
        """
        pixel_dist, center = self.get_pixel_distance(bbox)
        real_dist = self.estimate_distance(bbox, object_class, 'hybrid')

        x1, y1, x2, y2 = bbox

        return {
            'pixel_distance': pixel_dist,
            'real_distance': real_dist,
            'bbox_center': center,
            'bbox_width': x2 - x1,
            'bbox_height': y2 - y1,
            'object_class': object_class
        }

    def calibrate_focal_length(self, known_distance, known_width, measured_bbox_width):
        """
        校准焦距

        使用方法：
        1. 在已知距离处放置已知宽度的物体
        2. 测量检测框宽度
        3. 调用此函数计算焦距

        Args:
            known_distance: 已知距离（米）
            known_width: 物体实际宽度（米）
            measured_bbox_width: 测量的bbox宽度（像素）

        Returns:
            focal_length: 校准后的焦距
        """
        self.focal_length = (measured_bbox_width * known_distance) / known_width
        print(f"✓ 焦距已校准: {self.focal_length:.1f} pixels")
        return self.focal_length

    def add_custom_object(self, class_name, width, height):
        """
        添加自定义物体类别及其尺寸

        Args:
            class_name: 类别名称
            width: 真实宽度（米）
            height: 真实高度（米）
        """
        self.object_dimensions[class_name] = {
            'width': width,
            'height': height
        }
        print(f"✓ 添加自定义类别: {class_name} (宽={width}m, 高={height}m)")


# 使用示例
if __name__ == "__main__":
    # 创建估算器
    estimator = DistanceEstimator(
        focal_length=1000,
        image_width=1280,
        image_height=720
    )

    # 示例1：估算一辆车的距离
    car_bbox = [100, 300, 250, 450]  # [x1, y1, x2, y2]
    distance = estimator.estimate_distance(car_bbox, 'car')
    print(f"车辆距离: {distance:.2f}米")

    # 示例2：估算一个人的距离
    person_bbox = [500, 350, 550, 500]
    distance = estimator.estimate_distance(person_bbox, 'person')
    print(f"行人距离: {distance:.2f}米")

    # 示例3：获取完整信息
    info = estimator.get_distance_info(car_bbox, 'car')
    print(f"\n完整信息:")
    print(f"  真实距离: {info['real_distance']:.2f}m")
    print(f"  像素距离: {info['pixel_distance']:.1f}px")
    print(f"  检测框尺寸: {info['bbox_width']:.0f}x{info['bbox_height']:.0f}")
