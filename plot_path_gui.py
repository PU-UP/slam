#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def load_path_txt(path):
    df = pd.read_csv(path)
    # 兼容可能存在的空格
    df.columns = [c.strip() for c in df.columns]
    required = ["t","px","py","pz","vx","vy","vz","qw","qx","qy","qz"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")
    return df

def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} path.txt")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        sys.exit(1)

    df = load_path_txt(path)
    t = df["t"].to_numpy()
    px = df["px"].to_numpy()
    py = df["py"].to_numpy()
    pz = df["pz"].to_numpy()

    t_min, t_max = float(t.min()), float(t.max())
    # 初始窗口：全段
    win = [t_min, t_max]

    # ---------- 画布布局 ----------
    plt.figure(figsize=(10,6))
    ax_xy = plt.axes([0.08, 0.30, 0.54, 0.65])  # 左上：XY轨迹
    ax_z  = plt.axes([0.70, 0.30, 0.27, 0.65])  # 右上：Z-时间曲线

    ax_smin = plt.axes([0.08, 0.18, 0.84, 0.03]) # 下：最小时间滑块
    ax_smax = plt.axes([0.08, 0.13, 0.84, 0.03]) # 下：最大时间滑块

    ax_btn_reset  = plt.axes([0.08, 0.06, 0.12, 0.05])
    ax_btn_save   = plt.axes([0.22, 0.06, 0.12, 0.05])

    # 顶部文字栏
    ax_info = plt.axes([0.08, 0.95, 0.89, 0.03])
    ax_info.axis("off")
    info_text = ax_info.text(0.01, 0.5, "", va="center", ha="left")

    # ---------- 初始绘制 ----------
    def subset(t0, t1):
        mask = (t >= t0) & (t <= t1)
        return mask

    mask = subset(win[0], win[1])

    xy_line, = ax_xy.plot(px[mask], py[mask], linewidth=1.5)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, linestyle="--", alpha=0.4)
    ax_xy.set_title("XY Trajectory")

    z_line, = ax_z.plot(t[mask], pz[mask], linewidth=1.2)
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Z (m)")
    ax_z.grid(True, linestyle="--", alpha=0.4)
    ax_z.set_title("Z vs Time")

    # ---------- 滑块 ----------
    s_min = Slider(ax=ax_smin, label="t_min", valmin=t_min, valmax=t_max, valinit=win[0])
    s_max = Slider(ax=ax_smax, label="t_max", valmin=t_min, valmax=t_max, valinit=win[1])

    # ---------- 回调 ----------
    def update_plot(_):
        t0 = min(s_min.val, s_max.val)
        t1 = max(s_min.val, s_max.val)
        m  = subset(t0, t1)

        # 更新曲线数据
        xy_line.set_xdata(px[m]); xy_line.set_ydata(py[m])
        z_line.set_xdata(t[m]);   z_line.set_ydata(pz[m])

        # 自适应边界
        if np.any(m):
            ax_xy.relim(); ax_xy.autoscale_view()
            ax_z.relim();  ax_z.autoscale_view()

        # 更新信息栏
        n = int(np.sum(m))
        info_text.set_text(f"Window: [{t0:.3f}, {t1:.3f}]  |  Frames: {n}")

        plt.draw()

    s_min.on_changed(update_plot)
    s_max.on_changed(update_plot)

    # ---------- 按钮 ----------
    btn_reset = Button(ax_btn_reset, "重置窗口")
    btn_save  = Button(ax_btn_save,  "保存截图")

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
        print(f"已保存: {out_png}")
    btn_save.on_clicked(on_save)

    # 初始化信息栏
    info_text.set_text(f"Window: [{win[0]:.3f}, {win[1]:.3f}]  |  Frames: {np.sum(mask)}")

    plt.show()

if __name__ == "__main__":
    main()
