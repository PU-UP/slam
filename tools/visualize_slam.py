#!/usr/bin/env python3
"""
SLAM Visualization Script

This script visualizes the output from the SFM reconstruction process.
It can display camera trajectories, 3D points, and feature tracks.

Usage:
    python visualize_slam.py --output_dir ./output --prefix final_
    python visualize_slam.py --output_dir ./output --prefix intermediate_5_ --interactive
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SLAMVisualizer:
    def __init__(self, output_dir, prefix="final_"):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.poses_file = self.output_dir / f"{prefix}poses.csv"
        self.points_file = self.output_dir / f"{prefix}points.csv"
        self.tracks_file = self.output_dir / f"{prefix}tracks.csv"
        self.intrinsics_file = self.output_dir / f"{prefix}intrinsics.json"
        
        self.load_data()
        
    def load_data(self):
        """Load all the data files"""
        print(f"Loading data from {self.output_dir}")
        
        # Load camera poses
        if self.poses_file.exists():
            self.poses_df = pd.read_csv(self.poses_file)
            print(f"Loaded {len(self.poses_df)} camera poses")
        else:
            print(f"Warning: {self.poses_file} not found")
            self.poses_df = pd.DataFrame()
            
        # Load 3D points
        if self.points_file.exists():
            self.points_df = pd.read_csv(self.points_file)
            print(f"Loaded {len(self.points_df)} 3D points")
        else:
            print(f"Warning: {self.points_file} not found")
            self.points_df = pd.DataFrame()
            
        # Load feature tracks
        if self.tracks_file.exists():
            self.tracks_df = pd.read_csv(self.tracks_file)
            print(f"Loaded {len(self.tracks_df)} feature tracks")
        else:
            print(f"Warning: {self.tracks_file} not found")
            self.tracks_df = pd.DataFrame()
            
        # Load camera intrinsics
        if self.intrinsics_file.exists():
            with open(self.intrinsics_file, 'r') as f:
                self.intrinsics = json.load(f)
            print(f"Loaded camera intrinsics")
        else:
            print(f"Warning: {self.intrinsics_file} not found")
            self.intrinsics = None
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        qx, qy, qz, qw = q
        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
    
    def plot_trajectory_3d(self, save_path=None):
        """Plot 3D camera trajectory"""
        if self.poses_df.empty:
            print("No pose data available")
            return
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        positions = self.poses_df[['tx', 'ty', 'tz']].values
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera trajectory')
        
        # Plot camera orientations
        for i in range(0, len(self.poses_df), max(1, len(self.poses_df)//20)):
            row = self.poses_df.iloc[i]
            pos = np.array([row.tx, row.ty, row.tz])
            q = np.array([row.qx, row.qy, row.qz, row.qw])
            R = self.quaternion_to_rotation_matrix(q)
            
            # Draw camera axes
            scale = 0.1
            ax.quiver(pos[0], pos[1], pos[2], 
                     R[0, 0], R[0, 1], R[0, 2], 
                     length=scale, color='r', alpha=0.7)
            ax.quiver(pos[0], pos[1], pos[2], 
                     R[1, 0], R[1, 1], R[1, 2], 
                     length=scale, color='g', alpha=0.7)
            ax.quiver(pos[0], pos[1], pos[2], 
                     R[2, 0], R[2, 1], R[2, 2], 
                     length=scale, color='b', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Camera Trajectory')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot to {save_path}")
        
        plt.show()
    
    def plot_trajectory_2d(self, save_path=None):
        """Plot 2D camera trajectory (top view)"""
        if self.poses_df.empty:
            print("No pose data available")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract positions
        positions = self.poses_df[['tx', 'ty']].values
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Camera trajectory')
        
        # Plot camera orientations
        for i in range(0, len(self.poses_df), max(1, len(self.poses_df)//20)):
            row = self.poses_df.iloc[i]
            pos = np.array([row.tx, row.ty])
            q = np.array([row.qx, row.qy, row.qz, row.qw])
            R = self.quaternion_to_rotation_matrix(q)
            
            # Draw camera forward direction
            scale = 0.1
            ax.arrow(pos[0], pos[1], R[0, 2]*scale, R[1, 2]*scale, 
                    head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Camera Trajectory (Top View)')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 2D trajectory plot to {save_path}")
        
        plt.show()
    
    def plot_3d_points(self, save_path=None):
        """Plot 3D points"""
        if self.points_df.empty:
            print("No point data available")
            return
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract points
        points = self.points_df[['X', 'Y', 'Z']].values
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='red', s=1, alpha=0.6, label='3D points')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D points plot to {save_path}")
        
        plt.show()
    
    def plot_trajectory_and_points(self, save_path=None):
        """Plot both trajectory and points together"""
        if self.poses_df.empty or self.points_df.empty:
            print("Missing pose or point data")
            return
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        positions = self.poses_df[['tx', 'ty', 'tz']].values
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera trajectory')
        
        # Plot points
        points = self.points_df[['X', 'Y', 'Z']].values
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='red', s=1, alpha=0.6, label='3D points')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory and 3D Points')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined plot to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self):
        """Create interactive 3D plot using Plotly"""
        if self.poses_df.empty or self.points_df.empty:
            print("Missing pose or point data")
            return
        
        # Create figure
        fig = go.Figure()
        
        # Add camera trajectory
        positions = self.poses_df[['tx', 'ty', 'tz']].values
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            name='Camera Trajectory',
            line=dict(color='blue', width=4)
        ))
        
        # Add 3D points
        points = self.points_df[['X', 'Y', 'Z']].values
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name='3D Points',
            marker=dict(size=2, color='red', opacity=0.6)
        ))
        
        # Add camera frames
        for i in range(0, len(self.poses_df), max(1, len(self.poses_df)//10)):
            row = self.poses_df.iloc[i]
            pos = np.array([row.tx, row.ty, row.tz])
            q = np.array([row.qx, row.qy, row.qz, row.qw])
            R = self.quaternion_to_rotation_matrix(q)
            
            scale = 0.1
            # Add coordinate frame
            for axis, color in [(0, 'red'), (1, 'green'), (2, 'blue')]:
                end_pos = pos + R[:, axis] * scale
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], end_pos[0]],
                    y=[pos[1], end_pos[1]],
                    z=[pos[2], end_pos[2]],
                    mode='lines',
                    name=f'Camera {i} axis {axis}',
                    line=dict(color=color, width=3),
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Interactive SLAM Reconstruction',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Try to show in browser, with fallback for WSL
        try:
            # Check if we're in WSL environment
            import os
            if os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower():
                print("WSL environment detected. Saving HTML file instead of opening browser.")
                html_file = self.output_dir / "interactive_plot.html"
                fig.write_html(str(html_file))
                print(f"Interactive plot saved to: {html_file}")
                print("Please open this file in your Windows browser to view the interactive visualization.")
            else:
                fig.show()
        except Exception as e:
            print(f"Could not open browser: {e}")
            print("Saving HTML file instead...")
            html_file = self.output_dir / "interactive_plot.html"
            fig.write_html(str(html_file))
            print(f"Interactive plot saved to: {html_file}")
            print("Please open this file in your browser to view the interactive visualization.")
    
    def analyze_reconstruction(self):
        """Analyze the reconstruction quality"""
        print("\n=== Reconstruction Analysis ===")
        
        if not self.poses_df.empty:
            print(f"Number of camera poses: {len(self.poses_df)}")
            
            # Calculate trajectory length
            positions = self.poses_df[['tx', 'ty', 'tz']].values
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
            print(f"Total trajectory length: {total_distance:.3f}m")
            print(f"Average frame-to-frame distance: {np.mean(distances):.3f}m")
            
            # Calculate bounding box
            print(f"Bounding box:")
            print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
            print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
            print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        if not self.points_df.empty:
            print(f"Number of 3D points: {len(self.points_df)}")
            
            # Point statistics
            points = self.points_df[['X', 'Y', 'Z']].values
            print(f"Point cloud bounding box:")
            print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        if not self.tracks_df.empty:
            print(f"Number of feature tracks: {len(self.tracks_df)}")
            
            # Track statistics
            track_lengths = self.tracks_df.groupby('landmark_id').size()
            print(f"Average track length: {track_lengths.mean():.2f} frames")
            print(f"Max track length: {track_lengths.max()} frames")
            print(f"Min track length: {track_lengths.min()} frames")


def main():
    parser = argparse.ArgumentParser(description='SLAM Visualization Tool')
    parser.add_argument('--output_dir', type=str, default='../build/output',
                        help='Output directory containing reconstruction results')
    parser.add_argument('--prefix', type=str, default='final_',
                        help='Prefix for output files (e.g., "final_", "intermediate_5_")')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Show 3D trajectory plot')
    parser.add_argument('--plot_2d', action='store_true',
                        help='Show 2D trajectory plot')
    parser.add_argument('--plot_points', action='store_true',
                        help='Show 3D points plot')
    parser.add_argument('--plot_combined', action='store_true',
                        help='Show combined trajectory and points plot')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive 3D plot')
    parser.add_argument('--analyze', action='store_true',
                        help='Show reconstruction analysis')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SLAMVisualizer(args.output_dir, args.prefix)
    
    # Show analysis
    if args.analyze or not any([args.plot_3d, args.plot_2d, args.plot_points, args.plot_combined, args.interactive]):
        visualizer.analyze_reconstruction()
    
    # Show plots
    if args.plot_3d:
        save_path = f"{args.output_dir}/trajectory_3d.png" if args.save_plots else None
        visualizer.plot_trajectory_3d(save_path)
    
    if args.plot_2d:
        save_path = f"{args.output_dir}/trajectory_2d.png" if args.save_plots else None
        visualizer.plot_trajectory_2d(save_path)
    
    if args.plot_points:
        save_path = f"{args.output_dir}/points_3d.png" if args.save_plots else None
        visualizer.plot_3d_points(save_path)
    
    if args.plot_combined:
        save_path = f"{args.output_dir}/combined.png" if args.save_plots else None
        visualizer.plot_trajectory_and_points(save_path)
    
    if args.interactive:
        visualizer.create_interactive_plot()


if __name__ == "__main__":
    main()