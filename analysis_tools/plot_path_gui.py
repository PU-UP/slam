#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def load_path_txt(path):
    df = pd.read_csv(path)
    # Handle possible spaces in column names
    df.columns = [c.strip() for c in df.columns]
    required = ["t","px","py","pz","vx","vy","vz","qw","qx","qy","qz"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return df

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path.txt")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    df = load_path_txt(path)
    t = df["t"].to_numpy()
    px = df["px"].to_numpy()
    py = df["py"].to_numpy()
    pz = df["pz"].to_numpy()
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    vz = df["vz"].to_numpy()
    qw = df["qw"].to_numpy()
    qx = df["qx"].to_numpy()
    qy = df["qy"].to_numpy()
    qz = df["qz"].to_numpy()
    
    # Convert quaternions to Euler angles (in degrees for better readability)
    roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # Calculate velocities
    v_xy = np.sqrt(vx**2 + vy**2)  # XY plane velocity magnitude

    t_min, t_max = float(t.min()), float(t.max())
    # Initial window: full range
    win = [t_min, t_max]

    # ---------- Canvas layout ----------
    plt.figure(figsize=(16,10))
    # Top row: XY trajectory, Z-time, Angles
    ax_xy = plt.axes([0.05, 0.55, 0.28, 0.40])     # Top left: XY trajectory
    ax_z  = plt.axes([0.36, 0.55, 0.28, 0.40])     # Top center: Z-time curve
    ax_angles = plt.axes([0.67, 0.55, 0.28, 0.40]) # Top right: Angles vs time
    
    # Bottom row: XY velocity, Z velocity
    ax_v_xy = plt.axes([0.05, 0.25, 0.42, 0.25])   # Bottom left: XY velocity
    ax_v_z  = plt.axes([0.53, 0.25, 0.42, 0.25])   # Bottom right: Z velocity

    ax_smin = plt.axes([0.05, 0.15, 0.90, 0.03])   # Bottom: min time slider
    ax_smax = plt.axes([0.05, 0.10, 0.90, 0.03])   # Bottom: max time slider

    ax_btn_reset  = plt.axes([0.05, 0.03, 0.12, 0.05])
    ax_btn_save   = plt.axes([0.20, 0.03, 0.12, 0.05])

    # Top text bar
    ax_info = plt.axes([0.05, 0.97, 0.90, 0.02])
    ax_info.axis("off")
    info_text = ax_info.text(0.01, 0.5, "", va="center", ha="left")

    # ---------- Initial plotting ----------
    def subset(t0, t1):
        mask = (t >= t0) & (t <= t1)
        return mask

    mask = subset(win[0], win[1])

    # XY trajectory plot
    xy_line, = ax_xy.plot(px[mask], py[mask], linewidth=1.5)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, linestyle="--", alpha=0.4)
    ax_xy.set_title("XY Trajectory")

    # Z vs time plot
    z_line, = ax_z.plot(t[mask], pz[mask], linewidth=1.2)
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Z (m)")
    ax_z.grid(True, linestyle="--", alpha=0.4)
    ax_z.set_title("Z vs Time")
    
    # Angles vs time plot
    roll_line, = ax_angles.plot(t[mask], roll_deg[mask], 'r-', linewidth=1.2, label='Roll')
    pitch_line, = ax_angles.plot(t[mask], pitch_deg[mask], 'g-', linewidth=1.2, label='Pitch')
    yaw_line, = ax_angles.plot(t[mask], yaw_deg[mask], 'b-', linewidth=1.2, label='Yaw')
    ax_angles.set_xlabel("Time (s)")
    ax_angles.set_ylabel("Angle (deg)")
    ax_angles.grid(True, linestyle="--", alpha=0.4)
    ax_angles.set_title("Roll, Pitch, Yaw vs Time")
    ax_angles.legend()
    
    # XY velocity vs time plot
    v_xy_line, = ax_v_xy.plot(t[mask], v_xy[mask], 'purple', linewidth=1.2)
    ax_v_xy.set_xlabel("Time (s)")
    ax_v_xy.set_ylabel("XY Velocity (m/s)")
    ax_v_xy.grid(True, linestyle="--", alpha=0.4)
    ax_v_xy.set_title("XY Plane Velocity vs Time")
    
    # Z velocity vs time plot
    v_z_line, = ax_v_z.plot(t[mask], vz[mask], 'orange', linewidth=1.2)
    ax_v_z.set_xlabel("Time (s)")
    ax_v_z.set_ylabel("Z Velocity (m/s)")
    ax_v_z.grid(True, linestyle="--", alpha=0.4)
    ax_v_z.set_title("Z Velocity vs Time")

    # ---------- Sliders ----------
    s_min = Slider(ax=ax_smin, label="t_min", valmin=t_min, valmax=t_max, valinit=win[0])
    s_max = Slider(ax=ax_smax, label="t_max", valmin=t_min, valmax=t_max, valinit=win[1])

    # ---------- Callbacks ----------
    def update_plot(_):
        t0 = min(s_min.val, s_max.val)
        t1 = max(s_min.val, s_max.val)
        m  = subset(t0, t1)

        # Update curve data
        xy_line.set_xdata(px[m]); xy_line.set_ydata(py[m])
        z_line.set_xdata(t[m]);   z_line.set_ydata(pz[m])
        
        # Update angle plots
        roll_line.set_xdata(t[m]); roll_line.set_ydata(roll_deg[m])
        pitch_line.set_xdata(t[m]); pitch_line.set_ydata(pitch_deg[m])
        yaw_line.set_xdata(t[m]); yaw_line.set_ydata(yaw_deg[m])
        
        # Update velocity plots
        v_xy_line.set_xdata(t[m]); v_xy_line.set_ydata(v_xy[m])
        v_z_line.set_xdata(t[m]); v_z_line.set_ydata(vz[m])

        # Auto-scale bounds
        if np.any(m):
            ax_xy.relim(); ax_xy.autoscale_view()
            ax_z.relim();  ax_z.autoscale_view()
            ax_angles.relim(); ax_angles.autoscale_view()
            ax_v_xy.relim(); ax_v_xy.autoscale_view()
            ax_v_z.relim(); ax_v_z.autoscale_view()

        # Update info bar
        n = int(np.sum(m))
        info_text.set_text(f"Window: [{t0:.3f}, {t1:.3f}]  |  Frames: {n}")

        plt.draw()

    s_min.on_changed(update_plot)
    s_max.on_changed(update_plot)

    # ---------- Buttons ----------
    btn_reset = Button(ax_btn_reset, "Reset Window")
    btn_save  = Button(ax_btn_save,  "Save Screenshot")

    def on_reset(event):
        s_min.reset()
        s_max.reset()
    btn_reset.on_clicked(on_reset)

    def on_save(event):
        t0 = min(s_min.val, s_max.val)
        t1 = max(s_min.val, s_max.val)
        base, _ = os.path.splitext(os.path.basename(path))
        out_png = f"{base}_{t0:.3f}_{t1:.3f}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_png}")
    btn_save.on_clicked(on_save)

    # Initialize info bar
    info_text.set_text(f"Window: [{win[0]:.3f}, {win[1]:.3f}]  |  Frames: {np.sum(mask)}")

    plt.show()

if __name__ == "__main__":
    main()
