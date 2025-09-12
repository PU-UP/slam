#!/usr/bin/env python3
"""
测试特征提取器的简单脚本
"""

import os
import cv2
import numpy as np
import subprocess
import sys

def create_test_image(output_path, size=(640, 480)):
    """创建一个测试图像"""
    # 创建一个带有角点和纹理的测试图像
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 50
    
    # 添加一些几何图形
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (400, 200), 80, (200, 200, 200), -1)
    cv2.ellipse(img, (300, 350), (120, 60), 30, 0, 360, (180, 180, 180), -1)
    
    # 添加一些噪声纹理
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 添加一些角点
    corner_points = np.array([
        [100, 300], [200, 100], [500, 150], [450, 400], [250, 250]
    ])
    for point in corner_points:
        cv2.circle(img, tuple(point), 10, (255, 255, 255), -1)
    
    cv2.imwrite(output_path, img)
    print(f"Created test image: {output_path}")
    return output_path

def test_feature_extractors():
    """测试所有特征提取器"""
    # 创建测试图像
    test_image = "test_image.png"
    create_test_image(test_image)
    
    # 构建项目
    print("\nBuilding project...")
    build_result = subprocess.run(["mkdir", "-p", "build"], capture_output=True)
    build_result = subprocess.run(["cd", "build", "&&", "cmake", "..", "&&", "make"], 
                                shell=True, capture_output=True, text=True)
    
    if build_result.returncode != 0:
        print("Build failed:")
        print(build_result.stderr)
        return False
    
    # 特征提取器类型
    feature_types = ["ORB", "ShiTomasi", "FAST_ORB"]
    
    # 创建debug输出目录
    os.makedirs("debug_output", exist_ok=True)
    
    # 测试每种特征提取器
    for feat_type in feature_types:
        print(f"\n{'='*50}")
        print(f"Testing {feat_type} feature extractor")
        print(f"{'='*50}")
        
        # 运行测试
        cmd = ["./build/test_feature_extractor", test_image, feat_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"{feat_type} test completed successfully!")
        else:
            print(f"{feat_type} test failed:")
            print(result.stderr)
    
    # 清理
    if os.path.exists(test_image):
        os.remove(test_image)
    
    print(f"\nDebug outputs saved to: debug_output/")
    return True

if __name__ == "__main__":
    success = test_feature_extractors()
    sys.exit(0 if success else 1)