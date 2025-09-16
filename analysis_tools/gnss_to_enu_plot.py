#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read gnss.txt (time, latitude, longitude, height, RTK status, ...),
convert LLH (WGS-84) to ENU (with reference point as origin), and plot.
- Default reference point: first fixed solution (status=4), or first record if not exists
- Use --ref-lat/--ref-lon/--ref-h to manually specify reference point
- Use --include-float to also plot float solutions (status=5)
- Output two plots: planar trajectory (E vs N) and E/N/U vs time curves
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


# WGS-84 常数
A = 6378137.0                     # 半长轴 a (m)
F = 1 / 298.257223563             # 扁率 f
E2 = F * (2 - F)                  # 第一偏心率平方 e^2

def geodetic_to_ecef(lat_deg: np.ndarray,
                     lon_deg: np.ndarray,
                     h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LLH -> ECEF (X, Y, Z)"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = A / np.sqrt(1.0 - E2 * sin_lat**2)  # 卯酉圈曲率半径
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1.0 - E2) + h) * sin_lat
    return X, Y, Z

def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                ref_lat_deg: float, ref_lon_deg: float, ref_h: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定参考点(ref_lat, ref_lon, ref_h)把 ECEF → ENU"""
    x0, y0, z0 = geodetic_to_ecef(np.array([ref_lat_deg]),
                                  np.array([ref_lon_deg]),
                                  np.array([ref_h]))
    dx = x - x0[0]
    dy = y - y0[0]
    dz = z - z0[0]

    lat0 = np.deg2rad(ref_lat_deg)
    lon0 = np.deg2rad(ref_lon_deg)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)

    # ECEF→ENU 旋转矩阵 R
    # [E N U]^T = R * [dX dY dZ]^T
    r11 = -sin_lon0;          r12 =  cos_lon0;          r13 = 0.0
    r21 = -sin_lat0*cos_lon0; r22 = -sin_lat0*sin_lon0; r23 = cos_lat0
    r31 =  cos_lat0*cos_lon0; r32 =  cos_lat0*sin_lon0; r33 = sin_lat0

    E = r11*dx + r12*dy + r13*dz
    N = r21*dx + r22*dy + r23*dz
    U = r31*dx + r32*dy + r33*dz
    return E, N, U

def choose_reference(lat, lon, h, status, manual_ref):
    """选择参考点：优先第一条固定解(status==4)，否则第一条；或使用手动指定"""
    if manual_ref is not None:
        return manual_ref
    idx_fixed = np.where(status == 4)[0]
    if idx_fixed.size > 0:
        i0 = idx_fixed[0]
    else:
        i0 = 0
    return float(lat[i0]), float(lon[i0]), float(h[i0])

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLH to ENU from gnss.txt and plot (WGS-84)")
    parser.add_argument("input", type=str, help="Input file path, e.g., gnss.txt")
    parser.add_argument("--include-float", action="store_true",
                        help="Include float solutions (status=5) in plot (layered with fixed solutions)")
    parser.add_argument("--ref-lat", type=float, default=None, help="Reference point latitude (deg)")
    parser.add_argument("--ref-lon", type=float, default=None, help="Reference point longitude (deg)")
    parser.add_argument("--ref-h",   type=float, default=None, help="Reference point height (m)")
    parser.add_argument("--time-unit", choices=["s","min"], default="s",
                        help="Time axis unit (relative to start): seconds s or minutes min")
    parser.add_argument("--save-prefix", type=str, default=None,
                        help="Save image prefix (if not set, only show window). E.g., --save-prefix out generates out_traj.png, out_timeseries.png")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Only take first 5 columns: time, lat, lon, h, status
    # Some rows may have extra columns, loadtxt only reads according to usecols
    data = np.loadtxt(path.as_posix(), usecols=(0,1,2,3,4))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t = data[:, 0]
    lat = data[:, 1]
    lon = data[:, 2]
    h   = data[:, 3]
    status = data[:, 4].astype(int)

    # Reference point
    manual_ref = None
    if args.ref_lat is not None and args.ref_lon is not None and args.ref_h is not None:
        manual_ref = (args.ref_lat, args.ref_lon, args.ref_h)
    ref_lat, ref_lon, ref_h = choose_reference(lat, lon, h, status, manual_ref)

    # Convert all to ECEF, then to ENU
    x, y, z = geodetic_to_ecef(lat, lon, h)
    E, N, U = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_h)

    # Time axis starts from 0 for easier reading
    t0 = t[0]
    t_rel = t - t0
    if args.time_unit == "min":
        t_rel = t_rel / 60.0

    # Masks
    mask_fix   = (status == 4)
    mask_float = (status == 5)

    # === Plot 1: Planar trajectory (E vs N) ===
    plt.figure(figsize=(7, 7))
    if np.any(mask_fix):
        plt.scatter(E[mask_fix], N[mask_fix], s=8, label="Fixed solution (status=4)", alpha=0.9)
    if args.include_float and np.any(mask_float):
        plt.scatter(E[mask_float], N[mask_float], s=8, label="Float solution (status=5)", alpha=0.6)
    plt.scatter([0],[0], marker="x", s=60, label=f"Reference point\n({ref_lat:.7f}°, {ref_lon:.7f}°, {ref_h:.3f} m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("ENU Planar Trajectory (E vs N)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    if args.save_prefix:
        out1 = f"{args.save_prefix}_traj.png"
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"Saved: {out1}")

    # === Plot 2: E/N/U vs time ===
    fig2, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    unit = "s" if args.time_unit == "s" else "min"
    def plot_series(ax, y, ylabel):
        if np.any(mask_fix):
            ax.plot(t_rel[mask_fix], y[mask_fix], linewidth=1.1, label="Fixed solution (4)")
        if args.include_float and np.any(mask_float):
            ax.plot(t_rel[mask_float], y[mask_float], linewidth=1.1, linestyle="--", label="Float solution (5)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    plot_series(axs[0], E, "E (m)")
    plot_series(axs[1], N, "N (m)")
    plot_series(axs[2], U, "U (m)")
    axs[2].set_xlabel(f"Relative time ({unit})")
    axs[0].legend(loc="best")

    fig2.suptitle("ENU Components vs Time")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.save_prefix:
        out2 = f"{args.save_prefix}_timeseries.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved: {out2}")

    plt.show()

if __name__ == "__main__":
    main()
