#!/usr/bin/env python3
"""
子图融合实现
功能：
1. 加载子图
2. 融合到全局地图
3. 可视化和保存结果
"""

import os
import sys
import struct
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 配置 matplotlib 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def decode_key(key: int) -> tuple:
    """完全匹配C++实现的key解码
    C++: 
    int32_t gi = int32_t(key >> 32);
    int32_t gj = int32_t(key & 0xFFFFFFFF);
    """
    # 提取高32位作为有符号整数 → gi
    gi = np.int32(key >> 32).item()
    # 提取低32位作为无符号数，然后转换为有符号整数 → gj
    low = key & 0xFFFFFFFF
    if low >= (1 << 31):
        gj = int(low - (1 << 32))
    else:
        gj = int(low)
    return int(gi), int(gj)

def encode_key(gi: int, gj: int) -> int:
    """完全匹配C++实现的key编码
    C++: return (int64_t(gi) << 32) | uint32_t(gj);
    """
    return (int(gi) << 32) | (int(gj) & 0xFFFFFFFF)

def log_odds(p: float) -> float:
    """完全匹配C++实现"""
    return np.log(p / (1.0 - p))

def clamp_log_odds(l: float) -> float:
    """完全匹配C++实现"""
    L_MAX = 20.0
    return np.clip(l, -L_MAX, L_MAX)

def inv_log_odds(l: float) -> float:
    """完全匹配C++实现"""
    return 1.0 / (1.0 + np.exp(-l))

class GridMap:
    """完全匹配C++的GridMap类"""
    def __init__(self):
        self.occ_map = {}  # Dict[int, float] - key: encoded (gi,gj), value: probability
        self.min_i = 0
        self.max_i = 0
        self.min_j = 0
        self.max_j = 0
        self.initialized = False
    
    def update_bounds(self, i: int, j: int):
        """更新地图边界"""
        if not self.initialized:
            self.min_i = i
            self.max_i = i
            self.min_j = j
            self.max_j = j
            self.initialized = True
        else:
            self.min_i = min(self.min_i, i)
            self.max_i = max(self.max_i, i)
            self.min_j = min(self.min_j, j)
            self.max_j = max(self.max_j, j)
    
    def update_occ(self, i: int, j: int, p_meas: float):
        """完全匹配C++的概率更新实现"""
        key = encode_key(i, j)
        eps = 1e-3
        
        # 1. 获取旧概率
        p_old = 0.5  # 默认先验
        if key in self.occ_map:
            p_old = np.clip(self.occ_map[key], eps, 1.0 - eps)
        
        # 限制测量概率范围
        p_meas = np.clip(p_meas, eps, 1.0 - eps)
        
        # 2. 计算log odds并相加
        l_old = log_odds(p_old)
        l_meas = log_odds(p_meas)
        l_new = clamp_log_odds(l_old + l_meas)
        
        # 3. 转回概率
        p_new = inv_log_odds(l_new)
        
        # 4. 更新地图
        was_present = key in self.occ_map
        self.occ_map[key] = p_new
        
        # 5. 更新边界
        if not was_present:
            if len(self.occ_map) == 1:
                self.min_i = self.max_i = i
                self.min_j = self.max_j = j
                self.initialized = True
            else:
                self.update_bounds(i, j)
    
    def to_matrix(self) -> np.ndarray:
        """转换为矩阵形式，用于可视化"""
        if not self.initialized:
            return np.full((1, 1), 0.5)
        
        h = self.max_i - self.min_i + 1
        w = self.max_j - self.min_j + 1
        grid = np.full((h, w), 0.5)
        
        for key, p in self.occ_map.items():
            gi, gj = decode_key(key)
            ii = gi - self.min_i
            jj = gj - self.min_j
            if 0 <= ii < h and 0 <= jj < w:
                grid[ii, jj] = p
        
        return grid

def load_submap(bin_path: str) -> tuple:
    """完全匹配C++的子图加载实现"""
    with open(bin_path, 'rb') as f:
        # 1) submap_id (int32)
        raw = f.read(4)
        if len(raw) < 4:
            raise RuntimeError(f"{bin_path} is too short (missing submap_id)")
        submap_id = struct.unpack('i', raw)[0]

        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError(f"{bin_path} is too short (missing ts).")
        ts = struct.unpack('d', raw)[0]
        
        # 2) first_pose (16 doubles)
        raw = f.read(8 * 16)
        if len(raw) < 8 * 16:
            raise RuntimeError(f"{bin_path} is too short (missing first_pose)")
        pose_vals = struct.unpack('d' * 16, raw)
        # 使用列优先顺序(order='F')来匹配Eigen的存储方式
        first_pose = np.array(pose_vals, dtype=np.float64).reshape((4, 4), order='F')
        
        # 3) bounds: min_i, max_i, min_j, max_j (4×int32)
        raw = f.read(4 * 4)
        if len(raw) < 16:
            raise RuntimeError(f"{bin_path} is too short (missing bounds)")
        min_i, max_i, min_j, max_j = struct.unpack('i' * 4, raw)
        
        # 4) occ_map size (uint64)
        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError(f"{bin_path} is too short (missing map_size)")
        map_size = struct.unpack('Q', raw)[0]
        
        # 5) Read map_size entries of (int64 key, double prob)
        occ_map = {}
        for _ in range(map_size):
            entry = f.read(8 + 8)
            if len(entry) < 16:
                raise RuntimeError(f"{bin_path} is too short (incomplete map entries)")
            key, prob = struct.unpack('q d', entry)
            occ_map[key] = prob
            
    return submap_id, ts,first_pose, min_i, max_i, min_j, max_j, occ_map

def visualize_map(occ_map: np.ndarray, p_free: float = 0.3, p_occ: float = 0.7):
    """可视化占用栅格地图"""
    vis = np.zeros((*occ_map.shape, 3), dtype=np.uint8)
    
    # 空闲区域显示为白色
    free_mask = occ_map < p_free
    vis[free_mask] = [255, 255, 255]
    
    # 未知区域显示为灰色
    unknown_mask = (occ_map >= p_free) & (occ_map <= p_occ)
    vis[unknown_mask] = [128, 128, 128]
    
    # 占用区域显示为黑色
    occ_mask = occ_map > p_occ
    vis[occ_mask] = [0, 0, 0]
    
    return vis

def fuse_submaps(folder_path: str, save_path: str = None):
    """融合所有子图"""
    # 创建全局地图
    global_map = GridMap()
    global_res = 0.1  # 全局地图分辨率
    submap_res = 0.05  # 子图分辨率
    
    # 获取所有子图文件并创建ID到文件路径的映射
    submap_files = glob.glob(os.path.join(folder_path, 'submap_*.bin'))
    if not submap_files:
        print(f"错误：在{folder_path}中没有找到子图文件")
        return None, None
    
    # 创建submap_id到文件路径的映射
    id_to_file = {}
    for bin_path in submap_files:
        try:
            submap_id = int(os.path.basename(bin_path).split('_')[1].split('.')[0])
            id_to_file[submap_id] = bin_path
        except (IndexError, ValueError):
            print(f"警告：无法从{bin_path}提取子图ID")
            continue
    
    if not id_to_file:
        print("错误：没有有效的子图文件")
        return None, None
    
    print(f"找到{len(id_to_file)}个子图文件")
    
    # 按ID顺序处理子图
    max_id = max(id_to_file.keys())
    for submap_id in range(max_id + 1):
        if submap_id not in id_to_file:
            print(f"警告：缺少submap_{submap_id}")
            continue
        
        bin_path = id_to_file[submap_id]
        print(f"\nProcessing submap_{submap_id}...")
        
        # 加载子图
        _, ts, first_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(bin_path)
        print(f"子图边界: i[{min_i}, {max_i}], j[{min_j}, {max_j}]")
        print(f"栅格数量: {len(occ_map)}")
        
        # 遍历子图中的每个占用栅格
        for key, p_meas in occ_map.items():
            # 1) 解码子图栅格索引
            sub_i, sub_j = decode_key(key)
            
            # 2) 计算子图坐标系下的物理坐标
            p_s = np.array([
                sub_i * submap_res,
                sub_j * submap_res,
                0.0
            ])
            
            # 3) 转换到世界坐标系
            p_w = first_pose[:3, :3] @ p_s + first_pose[:3, 3]
            
            # 4) 计算全局地图栅格索引
            gi_glob = int(np.floor(p_w[0] / global_res))
            gj_glob = int(np.floor(p_w[1] / global_res))
            
            # 5) 更新全局地图
            global_map.update_occ(gi_glob, gj_glob, p_meas)
    
    # 打印全局地图信息
    print(f"\n全局地图信息:")
    print(f"栅格数量: {len(global_map.occ_map)} cells")
    print(f"边界: i[{global_map.min_i}, {global_map.max_i}], j[{global_map.min_j}, {global_map.max_j}]")
    print(f"物理范围: x[{global_map.min_i*global_res:.2f}, {global_map.max_i*global_res:.2f}]")
    print(f"          y[{global_map.min_j*global_res:.2f}, {global_map.max_j*global_res:.2f}]")
    
    # 转换为矩阵形式
    grid = global_map.to_matrix()
    vis = visualize_map(grid) # 无论是否保存都生成可视化图像

    # 保存全局地图
    if save_path:
        # 保存为二进制文件
        save_global_map(global_map, save_path + '.bin')
        # 保存可视化结果
        plt.imsave(save_path + '.png', vis, cmap='gray')
        print(f"全局地图已保存到: {save_path}.bin 和 {save_path}.png")
    
    return global_map, vis # 返回地图和可视化图像

def save_global_map(global_map: GridMap, save_path: str):
    """保存全局地图为二进制文件"""
    with open(save_path, 'wb') as f:
        # 1. 保存地图边界
        f.write(struct.pack('i' * 4, global_map.min_i, global_map.max_i, 
                          global_map.min_j, global_map.max_j))
        
        # 2. 保存地图大小
        f.write(struct.pack('Q', len(global_map.occ_map)))
        
        # 3. 保存地图数据
        for key, prob in global_map.occ_map.items():
            f.write(struct.pack('q d', key, prob))

def load_global_map(load_path: str) -> GridMap:
    """加载全局地图"""
    global_map = GridMap()
    
    with open(load_path, 'rb') as f:
        # 1. 读取地图边界
        raw = f.read(4 * 4)
        min_i, max_i, min_j, max_j = struct.unpack('i' * 4, raw)
        global_map.min_i = min_i
        global_map.max_i = max_i
        global_map.min_j = min_j
        global_map.max_j = max_j
        global_map.initialized = True
        
        # 2. 读取地图大小
        raw = f.read(8)
        map_size = struct.unpack('Q', raw)[0]
        
        # 3. 读取地图数据
        for _ in range(map_size):
            raw = f.read(16)
            key, prob = struct.unpack('q d', raw)
            global_map.occ_map[key] = prob
    
    return global_map

# 添加子图匹配相关的函数
def add_noise_to_pose(pose: np.ndarray, translation_noise: float = 0.1, rotation_noise_deg: float = 5.0) -> np.ndarray:
    """给位姿添加噪声
    Args:
        pose: 4x4变换矩阵
        translation_noise: 平移噪声（米）
        rotation_noise_deg: 旋转噪声（度）
    """
    noisy_pose = pose.copy()
    
    # 添加平移噪声
    noisy_pose[:3, 3] += np.random.normal(0, translation_noise, 3)
    
    # 添加旋转噪声（简单起见，只在yaw方向添加）
    theta = np.random.normal(0, np.deg2rad(rotation_noise_deg))
    c, s = np.cos(theta), np.sin(theta)
    R_noise = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    noisy_pose[:3, :3] = R_noise @ pose[:3, :3]
    
    return noisy_pose

def compute_score(submap: GridMap, global_map: GridMap, pose: np.ndarray, 
                 submap_res: float = 0.05, global_res: float = 0.1) -> float:
    """计算子图在给定位姿下与全局地图的匹配得分"""
    score = 0
    count = 0
    
    # 遍历子图中的每个占用栅格
    for key, p_sub in submap.occ_map.items():
        if p_sub < 0.7:  # 只考虑占用概率高的点
            continue
            
        # 解码子图栅格索引
        sub_i, sub_j = decode_key(key)
        
        # 计算子图坐标系下的物理坐标
        p_s = np.array([
            sub_i * submap_res,
            sub_j * submap_res,
            0.0
        ])
        
        # 转换到世界坐标系
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # 计算全局地图栅格索引
        gi_glob = int(np.floor(p_w[0] / global_res))
        gj_glob = int(np.floor(p_w[1] / global_res))
        
        # 在全局地图中查找对应栅格
        key_glob = encode_key(gi_glob, gj_glob)
        if key_glob in global_map.occ_map:
            p_glob = global_map.occ_map[key_glob]
            score += abs(p_glob - p_sub)
            count += 1
    
    return score / count if count > 0 else float('inf')

def match_submap_to_global(submap_id: int, folder_path: str, global_map: GridMap,
                          max_iter: int = 100, visualize: bool = True):
    """将指定的子图与全局地图进行匹配"""
    # 加载子图
    bin_path = os.path.join(folder_path, f'submap_{submap_id}.bin')
    _, ts, true_pose, min_i, max_i, min_j, max_j, occ_map = load_submap(bin_path)
    
    # 创建子图的GridMap对象
    submap = GridMap()
    for key, prob in occ_map.items():
        gi, gj = decode_key(key)
        submap.update_occ(gi, gj, prob)
    
    # 添加初始噪声
    init_pose = add_noise_to_pose(true_pose)
    best_pose = init_pose.copy()
    best_score = compute_score(submap, global_map, best_pose)
    
    # 可视化准备
    if visualize:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Submap {submap_id} Matching Process')
    
    # 迭代优化
    for iter in range(max_iter):
        # 添加小的随机扰动
        current_pose = add_noise_to_pose(best_pose, 0.05, 2.0)
        current_score = compute_score(submap, global_map, current_pose)
        
        if current_score < best_score:
            best_pose = current_pose.copy()
            best_score = current_score
            
            if visualize and iter % 10 == 0:
                # 清除之前的图像
                ax1.clear()
                ax2.clear()
                
                # 显示全局地图
                global_grid = global_map.to_matrix()
                ax1.imshow(visualize_map(global_grid), origin='upper')
                ax1.set_title('Global Map')
                
                # 显示当前匹配结果
                temp_map = GridMap()
                for key, p_meas in submap.occ_map.items():
                    sub_i, sub_j = decode_key(key)
                    p_s = np.array([sub_i * 0.05, sub_j * 0.05, 0.0])
                    p_w = best_pose[:3, :3] @ p_s + best_pose[:3, 3]
                    gi_glob = int(np.floor(p_w[0] / 0.1))
                    gj_glob = int(np.floor(p_w[1] / 0.1))
                    temp_map.update_occ(gi_glob, gj_glob, p_meas)
                
                match_grid = temp_map.to_matrix()
                ax2.imshow(visualize_map(match_grid), origin='upper')
                ax2.set_title(f'Matching Result (Iter {iter})')
                
                plt.pause(0.1)
    
    if visualize:
        plt.ioff()
        plt.show()
    
    # 计算与真值的误差
    trans_error = np.linalg.norm(best_pose[:3, 3] - true_pose[:3, 3])
    rot_error = np.arccos((np.trace(best_pose[:3, :3] @ true_pose[:3, :3].T) - 1) / 2)
    
    print(f"\n匹配结果:")
    print(f"平移误差: {trans_error:.3f}m")
    print(f"旋转误差: {np.rad2deg(rot_error):.3f}度")
    
    return best_pose, true_pose

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 fuse_submaps.py /path/to/submaps_folder")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    save_path = os.path.join(folder_path, 'global_map')
    
    # 1. 融合子图并保存
    global_map, vis = fuse_submaps(folder_path, save_path)
    
    if global_map and vis is not None:
        plt.figure(figsize=(10, 10)) # 创建一个新的图窗
        plt.imshow(vis, origin='upper')
        plt.title('全局地图')
        plt.axis('off') # 关闭坐标轴
    
    # 显示全局地图
    plt.show()

if __name__ == '__main__':
    main() 