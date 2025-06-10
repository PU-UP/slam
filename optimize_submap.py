#!/usr/bin/env python3
"""
子图位姿优化脚本
功能：
1. 加载已保存的全局地图
2. 随机选择一个子图并添加位姿扰动
3. 使用scan-to-map方式优化位姿
4. 可视化优化前后的结果
"""

import os
import sys
import numpy as np
# from matplotlib import font_manager
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fuse_submaps import (
    load_submap, load_global_map, GridMap, decode_key,
    visualize_map, add_noise_to_pose
)
from particle_filter_matcher import ParticleFilter, encode_key, match_submap_with_particle_filter
from typing import List, Tuple
import glob
import argparse

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def compute_error_and_jacobian(pose_params: np.ndarray, 
                             submap: GridMap, 
                             global_map: GridMap,
                             submap_res: float = 0.05,
                             global_res: float = 0.1) -> tuple:
    """计算误差和雅可比矩阵
    参数：
        pose_params: [x, y, theta] - SE(2)位姿参数
        submap: 待优化的子图
        global_map: 全局地图
    返回：
        error: 总误差（不匹配栅格数量）
        jacobian: 雅可比矩阵 [3,]
    """
    x, y, theta = pose_params
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = np.array([x, y, 0])
    
    total_error = 0.0
    jacobian = np.zeros(3)
    total_points = 0
    
    # 遍历子图中的占用栅格
    for key, p_sub in submap.occ_map.items():
        # 只考虑占用栅格
        if p_sub < 0.6:  # 非占用栅格
            continue
        else:
            p_sub = 1.0
            
        total_points += 1
        
        # 解码子图栅格索引
        sub_i, sub_j = decode_key(key)
        
        # 计算子图坐标系下的物理坐标
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = R @ p_s + t
        
        # 计算全局地图栅格索引
        gi_glob = int(np.floor(p_w[0] / global_res))
        gj_glob = int(np.floor(p_w[1] / global_res))
        
        # 在7x7邻域内搜索最近的占用栅格
        min_dist = float('inf')
        best_dx = 0
        best_dy = 0
        found_match = False
        
        search_range = 3  # 搜索范围为7x7（中心点±3）
        for di in range(-search_range, search_range + 1):
            for dj in range(-search_range, search_range + 1):
                ni = gi_glob + di
                nj = gj_glob + dj
                
                key_n = (ni << 32) | (nj & 0xFFFFFFFF)
                if key_n in global_map.occ_map:
                    p_n = global_map.occ_map[key_n]
                    if p_n >= 0.6:  # 找到占用栅格
                        dist = di*di + dj*dj
                        if dist < min_dist:
                            min_dist = dist
                            best_dx = di * global_res
                            best_dy = dj * global_res
                            if dist == 0:  # 完全匹配
                                found_match = True
                                break
            if found_match:
                break
        
        if found_match:
            continue  # 如果找到完全匹配，不计算误差和梯度
            
        # 计算误差（直接使用距离）
        if min_dist < float('inf'):
            dist = np.sqrt(min_dist) * global_res
            total_error += dist  # 直接使用距离作为误差
            
            # 计算梯度（使用固定步长）
            dx = best_dx / (global_res * np.sqrt(min_dist))  # 归一化方向
            dy = best_dy / (global_res * np.sqrt(min_dist))
            
            # 使用较大的固定步长
            step = 1.0
            jacobian[0] += step * dx
            jacobian[1] += step * dy
            # 旋转梯度也使用较大的步长
            jacobian[2] += step * (-p_s[0] * s + p_s[1] * c) * (dx * c + dy * s)
        else:
            # 如果在搜索范围内没有找到占用栅格，使用最大误差和固定梯度
            total_error += search_range * global_res  # 最大搜索距离作为误差
            
            # 使用中心位置的梯度
            dx = -1.0 if p_w[0] > gi_glob * global_res + global_res/2 else 1.0
            dy = -1.0 if p_w[1] > gj_glob * global_res + global_res/2 else 1.0
            
            # 使用较大的固定步长
            step = 1.0
            jacobian[0] += step * dx
            jacobian[1] += step * dy
            jacobian[2] += step * (-p_s[0] * s + p_s[1] * c) * (dx * c + dy * s)
    
    # 返回平均误差和归一化的雅可比矩阵
    error = total_error / max(total_points, 1)
    jacobian = jacobian / max(total_points, 1)
    return error, jacobian

def transform_submap_to_size(submap: GridMap, pose: np.ndarray, 
                           target_shape: Tuple[int, int],
                           submap_res: float = 0.05,
                           global_res: float = 0.1) -> np.ndarray:
    """将子图转换到指定尺寸的栅格地图
    
    Args:
        submap: 源子图
        pose: 变换位姿
        target_shape: 目标尺寸 (height, width)
        submap_res: 子图分辨率
        global_res: 全局地图分辨率
    
    Returns:
        转换后的栅格地图，尺寸与target_shape相同
    """
    result = np.full(target_shape, 0.5)  # 默认值0.5表示未知
    
    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        
        # 转换到物理坐标
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 转换到全局地图栅格坐标
        gi_glob = int(np.floor(p_w[0] / global_res))
        gj_glob = int(np.floor(p_w[1] / global_res))
        
        # 检查是否在目标尺寸范围内
        if 0 <= gi_glob < target_shape[0] and 0 <= gj_glob < target_shape[1]:
            result[gi_glob, gj_glob] = p_sub
    
    return result

def visualize_optimization_step(ax1, ax2, 
                              global_map: GridMap,
                              submap: GridMap,
                              particles: List['Particle'],
                              current_pose: np.ndarray,
                              iter_num: int,
                              error: float):
    """优化过程的可视化"""
    ax1.clear()
    ax2.clear()
    
    # 1. 获取全局地图
    global_grid = global_map.to_matrix()
    
    # 计算全局地图的物理范围
    global_res = 0.1  # 全局地图分辨率
    x_min = global_map.min_i * global_res
    x_max = global_map.max_i * global_res
    y_min = global_map.min_j * global_res
    y_max = global_map.max_j * global_res
    
    # 将概率值转换为更清晰的显示
    vis_global = np.zeros_like(global_grid)
    vis_global[global_grid > 0.6] = 1.0  # 占用栅格
    vis_global[global_grid < 0.4] = 0.3  # 空闲栅格
    
    # 显示全局地图
    ax1.imshow(vis_global, cmap='gray', origin='upper',
               extent=[y_min, y_max, x_max, x_min])  # 注意这里x和y的顺序
    
    # 绘制粒子，使用渐变色表示权重
    weights = np.array([p.weight for p in particles])
    max_weight = weights.max()
    if max_weight > 0:
        weights = weights / max_weight
    
    # 直接使用物理坐标绘制粒子
    xs = []
    ys = []
    dirs = []  # 方向
    valid_weights = []
    for p, w in zip(particles, weights):
        # 检查是否在显示范围内
        if x_min <= p.x <= x_max and y_min <= p.y <= y_max:
            xs.append(p.y)  # matplotlib中x对应y坐标
            ys.append(p.x)  # matplotlib中y对应x坐标
            dirs.append([np.cos(p.theta), np.sin(p.theta)])
            valid_weights.append(w)
    
    if xs:  # 如果有有效的粒子
        # 绘制粒子位置
        scatter = ax1.scatter(xs, ys, 
                            c=valid_weights,
                            cmap='hot',
                            s=30,
                            alpha=0.6)
        
        # 绘制粒子方向
        for x, y, d, w in zip(xs, ys, dirs, valid_weights):
            if w > 0.5:  # 只显示权重较大的粒子的方向
                ax1.arrow(x, y, 
                         d[1]*0.3, d[0]*0.3,  # 缩小箭头长度
                         head_width=0.1, 
                         head_length=0.1,
                         fc='red', 
                         ec='red',
                         alpha=0.6)
    
    # 2. 右图：叠加显示
    # 转换子图到全局地图尺寸
    submap_grid = transform_submap_to_size(submap, current_pose, global_grid.shape)
    
    # 创建RGB图像用于叠加显示
    overlay = np.zeros((*global_grid.shape, 3))
    
    # 设置全局地图为灰度背景
    overlay[..., 0] = vis_global
    overlay[..., 1] = vis_global
    overlay[..., 2] = vis_global
    
    # 将变换后的子图叠加为红色
    valid_mask = submap_grid > 0.6
    overlay[valid_mask, 0] = 1.0  # 红色通道
    overlay[valid_mask, 1] = 0.0
    overlay[valid_mask, 2] = 0.0
    
    # 显示叠加结果，使用相同的坐标范围
    ax2.imshow(overlay, origin='upper',
               extent=[y_min, y_max, x_max, x_min])
    
    # 设置标题
    ax1.set_title(f'粒子分布 (迭代次数 {iter_num})')
    ax2.set_title(f'匹配结果 (误差 {error:.3f})')
    
    # 添加图例
    ax1.text(0.02, 0.98, '粒子权重:', transform=ax1.transAxes, 
         verticalalignment='top', color='white')
    ax2.text(0.02, 0.98, '红色: 当前子图\n灰色: 全局地图',
         transform=ax2.transAxes, verticalalignment='top', color='white')

    
    # 设置坐标轴标签
    ax1.set_xlabel('Y (米)')
    ax1.set_ylabel('X (米)')
    ax2.set_xlabel('Y (米)')
    ax2.set_ylabel('X (米)')
    
    # 保持两个子图的显示范围一致
    ax1.set_xlim([y_min, y_max])
    ax1.set_ylim([x_max, x_min])  # 注意y轴方向
    ax2.set_xlim([y_min, y_max])
    ax2.set_ylim([x_max, x_min])
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

def optimize_submap_pose(submap: GridMap, 
                        global_map: GridMap,
                        init_pose: np.ndarray,
                        max_iter: int = 100,
                        use_particle_filter: bool = True,
                        visualize: bool = False) -> tuple:
    """使用梯度下降或粒子滤波优化子图位姿"""
    if use_particle_filter:
        print("使用粒子滤波进行优化...")
        # 初始散布：x, y方向±1m, 角度±15度
        spread_x_m = 1.0
        spread_y_m = 1.0
        spread_theta_rad = np.deg2rad(15.0)

        optimized_pose, final_error = match_submap_with_particle_filter(
            submap, global_map, init_pose,
            n_particles=100, # 减少粒子数量
            n_iterations=200, 
            visualize=visualize,
            spread=(spread_x_m, spread_y_m, spread_theta_rad), 
            global_res=0.1 
        )
        return optimized_pose, final_error
    else:
        # 使用原有的ICP方法
        best_pose = init_pose.copy()
        min_error = float('inf')
        no_improvement_count = 0
        max_no_improvement = 10
        
        current_pose = init_pose.copy()
        
        # ICP迭代
        for iter in range(max_iter):
            # 1. 转换子图特征到全局坐标系
            R = current_pose[:3, :3]
            t = current_pose[:3, 3]
            
            total_error = 0.0
            total_weight = 0.0
            H = np.zeros((2, 2))
            b = np.zeros(2)
            
            point_match_count = 0
            line_match_count = 0
            
            # 2. 处理点特征
            transformed_features = submap_features.copy()
            transformed_features[:, :2] = (R[:2, :2] @ submap_features[:, :2].T).T + t[:2]
            
            # 找到特征点匹配
            point_matches = find_feature_matches(
                transformed_features.tolist(),
                global_features.tolist(),
                max_dist=1.0  # 增大匹配距离
            )
            
            point_match_count = len(point_matches)
            
            if point_match_count >= 3:
                # 提取匹配点对
                p = np.array([m[3][:2] for m in point_matches])  # 源点
                q = np.array([m[4][:2] for m in point_matches])  # 目标点
                w = np.array([m[5] for m in point_matches])      # 权重
                
                # 计算误差和权重
                point_error = sum(d * w for _, _, d, _, _, w in point_matches)
                total_error += point_error
                total_weight += sum(w)
                
                # 计算加权质心
                p_mean = np.average(p, axis=0, weights=w)
                q_mean = np.average(q, axis=0, weights=w)
                
                # 累积H矩阵
                for i in range(len(p)):
                    p_centered = p[i] - p_mean
                    q_centered = q[i] - q_mean
                    H += w[i] * np.outer(p_centered, q_centered)
            
            # 3. 处理线段特征
            if len(submap_lines) > 0 and len(global_lines) > 0:
                # 转换子图线段
                transformed_lines = []
                for line in submap_lines:
                    # 转换线段端点
                    start = np.array([line[0], line[1], 0])
                    end = np.array([line[2], line[3], 0])
                    
                    t_start = (R @ start) + t
                    t_end = (R @ end) + t
                    
                    # 计算新的角度和长度
                    dx = t_end[0] - t_start[0]
                    dy = t_end[1] - t_start[1]
                    angle = np.arctan2(dy, dx)
                    length = np.sqrt(dx*dx + dy*dy)
                    
                    transformed_lines.append([
                        t_start[0], t_start[1],
                        t_end[0], t_end[1],
                        angle, length
                    ])
                
                # 匹配线段
                line_matches = match_line_segments(
                    transformed_lines,
                    global_lines,
                    max_dist=0.5,
                    max_angle=np.pi/4
                )
                
                line_match_count = len(line_matches)
                
                if line_match_count > 0:
                    # 计算线段匹配的误差和贡献
                    for i, j, dist, angle_diff in line_matches:
                        src = transformed_lines[i]
                        target = global_lines[j]
                        
                        # 线段中点
                        src_mid = np.array([(src[0] + src[2])/2, (src[1] + src[3])/2])
                        target_mid = np.array([(target[0] + target[2])/2, 
                                             (target[1] + target[3])/2])
                        
                        # 使用距离和角度差异计算权重
                        w = 3.0 * np.exp(-dist/0.5) * np.exp(-angle_diff/(np.pi/4))  # 增加线段权重
                        
                        # 累积误差
                        total_error += (dist + angle_diff * 0.5) * w
                        total_weight += w
                        
                        # 累积H矩阵（使用线段中点）
                        H += w * np.outer(src_mid - np.mean(src_mid), 
                                        target_mid - np.mean(target_mid))
            
            print(f"迭代 {iter}: 点匹配数={point_match_count}, 线段匹配数={line_match_count}, " 
                  f"总误差={total_error:.3f}, 总权重={total_weight:.3f}")
            
            if total_weight == 0:
                print("警告: 没有有效的特征匹配!")
                break
            
            # 计算平均误差
            avg_error = total_error / total_weight
            
            # 保存最佳结果
            if avg_error < min_error:
                min_error = avg_error
                best_pose = current_pose.copy()
                print(f"更新最佳位姿, 误差={min_error:.3f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= max_no_improvement:
                print("连续多次没有改进，提前结束优化")
                break
            
            # 计算最优旋转
            U, S, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T
            
            # 确保是正交矩阵
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = Vt.T @ U.T
            
            # 计算最优平移（使用所有特征的平均偏移）
            if total_weight > 0:
                t_opt = b / total_weight
            else:
                break
            
            # 更新位姿
            new_pose = np.eye(4)
            new_pose[:2, :2] = R_opt
            new_pose[:2, 3] = t_opt
            
            # 将新位姿与当前位姿组合
            current_pose = new_pose @ init_pose
        
        return best_pose, min_error

def compute_matching_error(submap: GridMap,
                         global_map: GridMap,
                         pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1) -> float:
    """Compute matching error between submap and global map"""
    total_error = 0
    count = 0
    
    for key, p_sub_raw in submap.occ_map.items():
        # 对子图概率进行二值化
        p_sub = 1.0 if p_sub_raw > 0.6 else 0.0
            
        # 只考虑子图中的占用栅格
        if p_sub == 0.0:
            continue
            
        # Get submap grid coordinates
        sub_i, sub_j = decode_key(key)
        
        # Convert to physical coordinates
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # Transform to world coordinates
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # Convert to global map grid coordinates
        gi_glob = int(np.floor(p_w[0] / global_res))
        gj_glob = int(np.floor(p_w[1] / global_res))
        
        # Check occupancy in global map
        key_glob = encode_key(gi_glob, gj_glob)
        
        p_glob = 0.0 # 默认全局地图该位置非占用
        if key_glob in global_map.occ_map:
            p_glob_raw = global_map.occ_map[key_glob]
            p_glob = 1.0 if p_glob_raw > 0.6 else 0.0
            
        # 计算残差（这里直接是二值化后的差异）
        total_error += abs(p_glob - p_sub) 
        count += 1
    
    return total_error / max(count, 1)

def transform_submap(submap: GridMap, pose: np.ndarray) -> GridMap:
    """使用给定位姿变换子图"""
    transformed = GridMap()
    
    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        
        # 转换到物理坐标
        p_s = np.array([
            sub_i * 0.05,  # submap_res
            sub_j * 0.05,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 转换到全局地图栅格坐标
        gi_glob = int(np.floor(p_w[0] / 0.1))  # global_res
        gj_glob = int(np.floor(p_w[1] / 0.1))
        
        transformed.update_occ(gi_glob, gj_glob, p_sub)
    
    return transformed

def visualize_optimization(global_map: GridMap,
                         submap: GridMap,
                         true_pose: np.ndarray,
                         init_pose: np.ndarray,
                         opt_pose: np.ndarray,
                         save_path: str = None):
    """可视化优化结果"""
    # 准备全局地图和子图
    global_grid = global_map.to_matrix()
    
    # 转换子图到全局坐标系
    def transform_submap(pose: np.ndarray) -> np.ndarray:
        grid = np.zeros_like(global_grid)
        for key, p_meas in submap.occ_map.items():
            if p_meas < 0.6:  # 只显示占用栅格
                continue
            sub_i, sub_j = decode_key(key)
            p_s = np.array([sub_i * 0.05, sub_j * 0.05, 0.0])
            p_w = pose[:3, :3] @ p_s + pose[:3, 3]
            gi_glob = int(np.floor(p_w[0] / 0.1))
            gj_glob = int(np.floor(p_w[1] / 0.1))
            if 0 <= gi_glob - global_map.min_i < grid.shape[0] and \
               0 <= gj_glob - global_map.min_j < grid.shape[1]:
                grid[gi_glob - global_map.min_i, 
                     gj_glob - global_map.min_j] = 1
        return grid
    
    # 生成三个位置的子图
    true_grid = transform_submap(true_pose)
    init_grid = transform_submap(init_pose)
    opt_grid = transform_submap(opt_pose)
    
    # 计算误差（占用栅格的不匹配率）
    def compute_error(pred_grid, true_grid):
        pred_points = np.sum(pred_grid > 0)
        if pred_points == 0:
            return 1.0  # 如果没有预测点，返回100%错误
        mismatch = np.sum((pred_grid > 0) & (true_grid == 0))
        return mismatch / pred_points
    
    init_error = compute_error(init_grid, true_grid)
    opt_error = compute_error(opt_grid, true_grid)
    
    # 创建可视化图像
    vis = np.zeros((*global_grid.shape, 3))
    
    # 1. 设置全局地图的灰度背景
    background = np.zeros_like(global_grid)
    background[global_grid <= 0.4] = 0.7  # 空闲为灰色
    background[global_grid >= 0.6] = 0.0  # 占用为黑色
    background[np.logical_and(global_grid > 0.4, global_grid < 0.6)] = 0.3  # 未知为深灰色
    
    # 2. 将灰度背景复制到三个通道
    for i in range(3):
        vis[..., i] = background
    
    # 3. 在背景上叠加彩色标记
    # 蓝色表示初始位置
    vis[init_grid > 0] = [0, 0, 1]  # 蓝色
    # 红色表示优化后位置
    vis[opt_grid > 0] = [1, 0, 0]   # 红色
    # 绿色表示真值位置
    vis[true_grid > 0] = [0, 1, 0]  # 绿色
    
    # 创建figure和axes，为图例留出空间
    fig = plt.figure(figsize=(15, 10))  # 加宽图形以容纳图例
    gs = plt.GridSpec(1, 2, width_ratios=[4, 1])  # 创建网格，左侧4份，右侧1份
    ax = fig.add_subplot(gs[0])  # 主图在左侧
    ax_legend = fig.add_subplot(gs[1])  # 图例在右侧
    ax_legend.axis('off')  # 关闭图例区域的坐标轴
    
    # 显示主图
    ax.imshow(vis)
    ax.set_title(f'优化结果\n栅格占用不匹配率: 优化前: {init_error*100:.1f}%, 优化后: {opt_error*100:.1f}%')
    
    # 添加图例到右侧
    legend_elements = [ 
        plt.Rectangle((0, 0), 1, 1, fc=[0, 0, 1], label='优化前'),
        plt.Rectangle((0, 0), 1, 1, fc=[1, 0, 0], label='优化后'),
        plt.Rectangle((0, 0), 1, 1, fc=[0, 1, 0], label='真值'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.7, 0.7, 0.7], label='空闲区域'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.3, 0.3, 0.3], label='未知区域'),
        plt.Rectangle((0, 0), 1, 1, fc=[0, 0, 0], label='占用区域'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center left', fontsize=12)
    
    # 计算位姿误差
    init_trans_error = np.linalg.norm(init_pose[:2, 3] - true_pose[:2, 3])
    init_rot_error = np.abs(np.arctan2(init_pose[1, 0], init_pose[0, 0]) - 
                           np.arctan2(true_pose[1, 0], true_pose[0, 0]))
    opt_trans_error = np.linalg.norm(opt_pose[:2, 3] - true_pose[:2, 3])
    opt_rot_error = np.abs(np.arctan2(opt_pose[1, 0], opt_pose[0, 0]) - 
                          np.arctan2(true_pose[1, 0], true_pose[0, 0]))
    
    # 用figtext在图片下方显示误差信息，不遮挡地图
    fig.subplots_adjust(bottom=0.18)  # 给下方留出空间
    info_text = (
        f'位姿误差 (相对于真值)：\n'
        f'优化前: {init_trans_error:.3f} 米, {np.rad2deg(init_rot_error):.1f}°    '
        f'优化后: {opt_trans_error:.3f} 米, {np.rad2deg(opt_rot_error):.1f}°'
    )
    fig.text(0.5, 0.05, info_text, ha='center', va='center', fontsize=13, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path)
        print(f"优化结果已保存到: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="子图位姿优化脚本")
    parser.add_argument("folder_path", type=str, help="包含子图和全局地图的文件夹路径")
    parser.add_argument("--plot", action="store_true", help="显示粒子滤波中间过程的可视化")
    args = parser.parse_args()

    folder_path = args.folder_path
    plot_intermediate = args.plot # 获取--plot参数的值

    global_map_path = os.path.join(folder_path, 'global_map.bin')
    
    # 1. 加载全局地图
    print("加载全局地图...")
    global_map = load_global_map(global_map_path)
    
    # 2. 随机选择一个子图
    submap_files = [f for f in os.listdir(folder_path) 
                    if f.startswith('submap_') and f.endswith('.bin')]
    if not submap_files:
        print("错误：没有找到子图文件")
        return
    
    target_file = np.random.choice(submap_files)
    submap_id = int(target_file.split('_')[1].split('.')[0])
    print(f"选择子图 {submap_id} 进行优化")
    
    # 3. 加载子图
    submap_path = os.path.join(folder_path, target_file)
    _, ts, true_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(submap_path)
    
    # 创建子图对象
    submap = GridMap()
    for key, prob in occ_map.items():
        gi, gj = decode_key(key)
        submap.update_occ(gi, gj, prob)
    
    # 4. 添加初始噪声
    init_pose = add_noise_to_pose(true_pose, 0.5, 10.0)  # 0.5m, 10度
    
    # 5. 优化位姿
    print("开始优化...")
    opt_pose, error = optimize_submap_pose(submap, global_map, init_pose, visualize=plot_intermediate)
    print(f"优化完成，最终误差: {error:.6f}")
    
    # 6. 直接显示结果
    visualize_optimization(
        global_map, submap, true_pose, init_pose, opt_pose, None  # 设置save_path为None以直接显示
    )
    plt.show()  # 确保图像显示出来

if __name__ == '__main__':
    main() 