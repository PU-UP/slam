import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.stats import norm
from fuse_submaps import GridMap, decode_key, encode_key
import matplotlib as mpl

# 配置 matplotlib 支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class Particle:
    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta  # yaw angle in radians
        self.weight = weight
        
    def to_matrix(self) -> np.ndarray:
        """Convert particle pose to 4x4 transformation matrix"""
        T = np.eye(4)
        c, s = np.cos(self.theta), np.sin(self.theta)
        T[:3, :3] = np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])
        T[:3, 3] = [self.x, self.y, 0]
        return T

class ParticleFilter:
    def __init__(self, 
                 n_particles: int = 100,
                 motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.05),  # (x, y, theta)
                 measurement_noise: float = 0.1):
        self.n_particles = n_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.particles: List[Particle] = []
        
    def initialize_particles(self, init_pose: np.ndarray, spread: Tuple[float, float, float]):
        """Initialize particles around initial pose with given spread"""
        self.particles = []
        
        # Extract initial position and orientation
        x = init_pose[0, 3]
        y = init_pose[1, 3]
        theta = np.arctan2(init_pose[1, 0], init_pose[0, 0])
        
        # Generate particles with Gaussian distribution
        for _ in range(self.n_particles):
            px = np.random.normal(x, spread[0])
            py = np.random.normal(y, spread[1])
            ptheta = np.random.normal(theta, spread[2])
            self.particles.append(Particle(px, py, ptheta, 1.0/self.n_particles))
            
    def predict(self, delta_pose: np.ndarray):
        """Move particles according to motion model with noise"""
        # Extract motion from delta pose
        dx = delta_pose[0, 3]
        dy = delta_pose[1, 3]
        dtheta = np.arctan2(delta_pose[1, 0], delta_pose[0, 0])
        
        for p in self.particles:
            # Apply motion with noise
            p.x += dx + np.random.normal(0, self.motion_noise[0])
            p.y += dy + np.random.normal(0, self.motion_noise[1])
            p.theta += dtheta + np.random.normal(0, self.motion_noise[2])
            p.theta = np.mod(p.theta + np.pi, 2*np.pi) - np.pi  # normalize angle
            
    def update_weights(self, 
                      submap: 'GridMap',
                      global_map: 'GridMap',
                      submap_res: float = 0.05,
                      global_res: float = 0.1):
        """Update particle weights based on map matching score"""
        total_weight = 0.0
        
        for particle in self.particles:
            # Convert particle pose to transformation matrix
            T = particle.to_matrix()
            score = 0
            count = 0
            
            # For each occupied cell in submap
            for key, p_sub_raw in submap.occ_map.items():
                # 对子图概率进行二值化
                p_sub_binary = 1.0 if p_sub_raw > 0.6 else 0.0

                # 只考虑子图中的占用栅格进行匹配
                if p_sub_binary == 0.0:
                    continue
                    
                # Get submap grid coordinates
                sub_i, sub_j = decode_key(key)
                
                # Convert to physical coordinates
                p_s = np.array([
                    sub_i * submap_res,
                    sub_j * submap_res,
                    0.0
                ])
                
                # Transform to world coordinates using particle pose
                p_w = T[:3, :3] @ p_s + T[:3, 3]
                
                # Convert to global map grid coordinates
                gi_glob = int(np.floor(p_w[0] / global_res))
                gj_glob = int(np.floor(p_w[1] / global_res))
                
                # Check occupancy in global map
                key_glob = encode_key(gi_glob, gj_glob)
                
                p_glob_binary = 0.0 # 默认全局地图该位置为非占用
                if key_glob in global_map.occ_map:
                    p_glob_raw = global_map.occ_map[key_glob]
                    p_glob_binary = 1.0 if p_glob_raw > 0.6 else 0.0
                    
                # 计算二值化后的概率差异
                diff = abs(p_glob_binary - p_sub_binary)
                score += diff
                count += 1
            
            # Update particle weight
            if count > 0:
                avg_score = score / count
                # Convert score to weight (lower score = higher weight)
                weight = np.exp(-avg_score / self.measurement_noise)
                particle.weight = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
                
    def resample(self):
        """Resample particles based on their weights"""
        # Systematic resampling
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        
        # Calculate cumulative sum of weights
        cumsum = np.cumsum([p.weight for p in self.particles])
        
        # Create new particle set
        new_particles = []
        i = 0
        for position in positions:
            while cumsum[i] < position:
                i += 1
            # Copy selected particle
            old_particle = self.particles[i]
            new_particles.append(Particle(
                old_particle.x,
                old_particle.y,
                old_particle.theta,
                1.0/self.n_particles
            ))
        
        self.particles = new_particles
        
    def get_estimated_pose(self) -> np.ndarray:
        """Get weighted average pose from particles"""
        if not self.particles:
            return np.eye(4)
            
        # Weighted average of position
        avg_x = sum(p.x * p.weight for p in self.particles)
        avg_y = sum(p.y * p.weight for p in self.particles)
        
        # For orientation, we need to handle circular mean
        cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles)
        sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles)
        avg_theta = np.arctan2(sin_sum, cos_sum)
        
        # Convert to transformation matrix
        T = np.eye(4)
        c, s = np.cos(avg_theta), np.sin(avg_theta)
        T[:3, :3] = np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])
        T[:3, 3] = [avg_x, avg_y, 0]
        return T

def match_submap_with_particle_filter(submap: 'GridMap',
                                    global_map: 'GridMap',
                                    init_pose: np.ndarray,
                                    n_particles: int = 500,
                                    n_iterations: int = 200,
                                    visualize: bool = True,
                                    spread: Tuple[float, float, float] = (0.5, 0.5, np.pi/6),
                                    global_res: float = 0.1) -> Tuple[np.ndarray, float]:
    """Match submap to global map using particle filter
    
    Args:
        submap: Source submap to match
        global_map: Target global map
        init_pose: Initial pose estimate (4x4 matrix)
        n_particles: Number of particles to use
        n_iterations: Number of iterations
        visualize: Whether to show visualization
        spread: Spread for initializing particles
        global_res: Global map resolution
        
    Returns:
        best_pose: Estimated pose (4x4 matrix)
        final_error: Final matching error
    """
    # Initialize particle filter
    pf = ParticleFilter(
        n_particles=n_particles,
        motion_noise=(0.05, 0.05, 0.02),  # 调整这些参数以满足您的需求
        measurement_noise=0.08
    )
    
    # Initialize particles around initial pose
    pf.initialize_particles(
        init_pose,
        spread=spread  # Use the provided spread argument
    )
    
    # Setup visualization
    if visualize:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Particle Filter Map Matching')
    
    best_pose = init_pose.copy()
    min_error = float('inf')

    no_improvement_threshold = 20 # 连续20次没有显著改善则停止
    error_tolerance = 0.001     # 误差改善小于此值认为没有显著改善
    no_improvement_count = 0
    
    for iter in range(n_iterations):
        # Predict (random walk for exploration)
        pf.predict(np.eye(4)) # Rely on internal motion_noise for spread
        
        # Update weights based on map matching
        pf.update_weights(submap, global_map)
        
        # Get current estimate
        current_pose = pf.get_estimated_pose()
        
        # Calculate current error
        error = compute_matching_error(submap, global_map, current_pose)
        
        # Update best pose if needed
        if error < min_error:
            if min_error - error > error_tolerance: # 显著改善
                no_improvement_count = 0
            else: # 改善不显著
                no_improvement_count += 1
            min_error = error
            best_pose = current_pose.copy()
            print(f"Iteration {iter}: New best pose found, error = {error:.3f}")
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= no_improvement_threshold:
            print(f"Early stopping due to no significant improvement for {no_improvement_threshold} iterations.")
            if visualize:
                plt.ioff() 
                plt.close(fig) 
            break

        # Visualize if requested
        if visualize and iter % 5 == 0:
            ax1.clear()
            ax2.clear()
            
            # Plot global map
            global_grid = global_map.to_matrix()
            ax1.imshow(global_grid, cmap='gray', origin='upper')
            
            # Plot particles
            # 将粒子世界坐标转换为全局地图矩阵的相对坐标
            # global_res (0.1)是全局地图分辨率
            particle_matrix_cols = [(p.y / global_res) - global_map.min_j for p in pf.particles]
            particle_matrix_rows = [(p.x / global_res) - global_map.min_i for p in pf.particles]
            weights = [p.weight * 1000 for p in pf.particles]  # 缩放权重以便可视化
            
            # scatter期望 (X, Y) 坐标，其中X是列索引，Y是行索引
            ax1.scatter(particle_matrix_cols, particle_matrix_rows, c='red', s=weights, alpha=0.5)
            
            # Plot transformed submap
            temp_map = transform_submap(submap, current_pose)
            match_grid = temp_map.to_matrix()
            ax2.imshow(match_grid, cmap='gray', origin='upper')
            ax2.set_title(f'Matching Result (Iter {iter})')
            
            plt.pause(0.1) # 只有在plot模式下才暂停
        
        # Resample particles
        pf.resample()
    
    return best_pose, min_error

def compute_matching_error(submap: 'GridMap',
                         global_map: 'GridMap',
                         pose: np.ndarray,
                         submap_res: float = 0.05,
                         global_res: float = 0.1) -> float:
    """Compute matching error between submap and global map"""
    total_error = 0
    count = 0
    
    for key, p_sub in submap.occ_map.items():
        if p_sub < 0.6:
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
        if key_glob in global_map.occ_map:
            p_glob = global_map.occ_map[key_glob]
            total_error += abs(p_glob - p_sub)
            count += 1
    
    return total_error / max(count, 1)

def transform_submap(submap: 'GridMap', pose: np.ndarray) -> 'GridMap':
    """Transform submap using given pose"""
    transformed = GridMap()
    
    for key, p_sub in submap.occ_map.items():
        sub_i, sub_j = decode_key(key)
        
        # Convert to physical coordinates
        p_s = np.array([
            sub_i * 0.05,  # submap_res
            sub_j * 0.05,
            0.0
        ])
        
        # Transform to world coordinates
        p_w = pose[:3, :3] @ p_s + pose[:3, 3]
        
        # Convert to global map grid coordinates
        gi_glob = int(np.floor(p_w[0] / 0.1))  # global_res
        gj_glob = int(np.floor(p_w[1] / 0.1))
        
        transformed.update_occ(gi_glob, gj_glob, p_sub)
    
    return transformed 