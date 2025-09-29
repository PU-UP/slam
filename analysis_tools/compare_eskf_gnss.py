#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot GNSS (ENU) trajectory together with ESKF result after alignment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# WGS-84 constants
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def geodetic_to_ecef(lat_deg: np.ndarray,
                     lon_deg: np.ndarray,
                     h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = A / np.sqrt(1.0 - E2 * sin_lat ** 2)
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1.0 - E2) + h) * sin_lat
    return X, Y, Z


def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                ref_lat_deg: float, ref_lon_deg: float, ref_h: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    r11 = -sin_lon0
    r12 = cos_lon0
    r13 = 0.0
    r21 = -sin_lat0 * cos_lon0
    r22 = -sin_lat0 * sin_lon0
    r23 = cos_lat0
    r31 = cos_lat0 * cos_lon0
    r32 = cos_lat0 * sin_lon0
    r33 = sin_lat0

    E = r11 * dx + r12 * dy + r13 * dz
    N = r21 * dx + r22 * dy + r23 * dz
    U = r31 * dx + r32 * dy + r33 * dz
    return E, N, U


def choose_reference(lat: np.ndarray,
                     lon: np.ndarray,
                     h: np.ndarray,
                     status: np.ndarray,
                     manual_ref: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if manual_ref is not None:
        return manual_ref
    idx_fixed = np.where(status == 4)[0]
    i0 = int(idx_fixed[0]) if idx_fixed.size > 0 else 0
    return float(lat[i0]), float(lon[i0]), float(h[i0])


@dataclass
class Trajectory:
    time: np.ndarray
    pos: np.ndarray  # shape (N, 3)
    info_mask: Optional[np.ndarray] = None  # e.g. fix mask for gnss


def load_gnss(path: Path,
              ref_lat: Optional[float],
              ref_lon: Optional[float],
              ref_h: Optional[float]) -> Trajectory:
    data = np.loadtxt(path.as_posix(), usecols=(0, 1, 2, 3, 4))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0]
    lat = data[:, 1]
    lon = data[:, 2]
    h = data[:, 3]
    status = data[:, 4].astype(int)
    manual_ref = None
    if ref_lat is not None and ref_lon is not None and ref_h is not None:
        manual_ref = (ref_lat, ref_lon, ref_h)
    ref_lat, ref_lon, ref_h = choose_reference(lat, lon, h, status, manual_ref)
    x, y, z = geodetic_to_ecef(lat, lon, h)
    E, N, U = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_h)
    pos = np.column_stack((E, N, U))
    mask_fix = (status == 4)
    return Trajectory(time=t, pos=pos, info_mask=mask_fix)


def load_eskf(path: Path) -> Trajectory:
    df = pd.read_csv(path)
    required = ["t", "px", "py", "pz"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"ESKF file missing column: {col}")
    t = df["t"].to_numpy()
    pos = df[["px", "py", "pz"]].to_numpy()
    return Trajectory(time=t, pos=pos)


def motion_start_time(traj: Trajectory, threshold: float) -> float:
    pos = traj.pos
    displacements = np.linalg.norm(pos - pos[0], axis=1)
    idx = np.argmax(displacements >= threshold)
    if displacements[idx] < threshold:
        # no motion above threshold
        return float(traj.time[0])
    return float(traj.time[idx])


def resample_positions(time_src: np.ndarray,
                       pos_src: np.ndarray,
                       time_query: np.ndarray) -> np.ndarray:
    if np.any(np.diff(time_src) <= 0):
        order = np.argsort(time_src)
        time_src = time_src[order]
        pos_src = pos_src[order]
    resampled = np.empty((time_query.size, pos_src.shape[1]))
    for i in range(pos_src.shape[1]):
        resampled[:, i] = np.interp(time_query, time_src, pos_src[:, i])
    return resampled


def align_rigid_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find rotation (2x2) and translation (2,) to align src onto dst."""
    if src_xy.shape != dst_xy.shape:
        raise ValueError("Source and destination must have same shape")
    mu_src = src_xy.mean(axis=0)
    mu_dst = dst_xy.mean(axis=0)
    src_centered = src_xy - mu_src
    dst_centered = dst_xy - mu_dst
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def main():
    parser = argparse.ArgumentParser(
        description="Align and compare GNSS (ENU) trajectory with ESKF result.")
    parser.add_argument("gnss", type=str, help="Path to gnss.txt")
    parser.add_argument("eskf", type=str, help="Path to eskf_result.txt")
    parser.add_argument("--ref-lat", type=float, default=None, help="Reference latitude for ENU")
    parser.add_argument("--ref-lon", type=float, default=None, help="Reference longitude for ENU")
    parser.add_argument("--ref-h", type=float, default=None, help="Reference height for ENU")
    parser.add_argument("--motion-threshold", type=float, default=2.0,
                        help="Motion threshold (meters) to determine start of movement")
    parser.add_argument("--align-duration", type=float, default=20.0,
                        help="Duration in seconds for alignment segment once motion starts")
    parser.add_argument("--align-samples", type=int, default=200,
                        help="Number of resampled points used for alignment")
    parser.add_argument("--save", type=str, default=None,
                        help="If set, save figure to this path instead of showing")

    args = parser.parse_args()

    gnss_path = Path(args.gnss)
    eskf_path = Path(args.eskf)
    if not gnss_path.exists():
        raise FileNotFoundError(f"GNSS file not found: {gnss_path}")
    if not eskf_path.exists():
        raise FileNotFoundError(f"ESKF file not found: {eskf_path}")

    gnss_traj = load_gnss(gnss_path, args.ref_lat, args.ref_lon, args.ref_h)
    eskf_traj = load_eskf(eskf_path)

    # Determine alignment window after motion begins in both trajectories
    gnss_motion_start = motion_start_time(gnss_traj, args.motion_threshold)
    eskf_motion_start = motion_start_time(eskf_traj, args.motion_threshold)
    align_t0 = max(gnss_motion_start, eskf_motion_start)
    align_t1 = align_t0 + args.align_duration
    max_possible = min(float(gnss_traj.time[-1]), float(eskf_traj.time[-1]))
    if align_t1 > max_possible:
        align_t1 = max_possible
    if align_t1 <= align_t0:
        raise RuntimeError("Not enough overlapping data after motion for alignment")

    time_align = np.linspace(align_t0, align_t1, args.align_samples)
    gnss_align = resample_positions(gnss_traj.time, gnss_traj.pos, time_align)
    eskf_align = resample_positions(eskf_traj.time, eskf_traj.pos, time_align)

    R_xy, t_xy = align_rigid_2d(eskf_align[:, :2], gnss_align[:, :2])
    eskf_xy_aligned = (R_xy @ eskf_traj.pos[:, :2].T).T + t_xy
    tz = float(np.mean(gnss_align[:, 2] - eskf_align[:, 2]))
    eskf_pos_aligned = np.column_stack((eskf_xy_aligned, eskf_traj.pos[:, 2] + tz))

    # Resample for error statistics using GNSS timestamps in overlap
    t_overlap_start = max(float(gnss_traj.time[0]), float(eskf_traj.time[0]))
    t_overlap_end = min(float(gnss_traj.time[-1]), float(eskf_traj.time[-1]))
    mask_gnss = (gnss_traj.time >= t_overlap_start) & (gnss_traj.time <= t_overlap_end)
    if not np.any(mask_gnss):
        raise RuntimeError("No overlapping time range between GNSS and ESKF data")
    t_eval = gnss_traj.time[mask_gnss]
    gnss_eval = gnss_traj.pos[mask_gnss]
    eskf_eval = resample_positions(eskf_traj.time, eskf_pos_aligned, t_eval)

    diff = eskf_eval - gnss_eval
    xy_error = np.linalg.norm(diff[:, :2], axis=1)
    z_error = diff[:, 2]
    rmse_xy = float(np.sqrt(np.mean(xy_error ** 2)))
    rmse_z = float(np.sqrt(np.mean(z_error ** 2)))
    rmse_3d = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    ax_xy = fig.add_subplot(gs[0, :])
    gnss_mask_fix = gnss_traj.info_mask if gnss_traj.info_mask is not None else np.ones_like(gnss_traj.time, dtype=bool)
    ax_xy.plot(gnss_traj.pos[gnss_mask_fix, 0], gnss_traj.pos[gnss_mask_fix, 1], 'o', markersize=2,
               label='GNSS (status=4)')
    if not np.all(gnss_mask_fix):
        ax_xy.plot(gnss_traj.pos[~gnss_mask_fix, 0], gnss_traj.pos[~gnss_mask_fix, 1], '.', markersize=2,
                   alpha=0.4, label='GNSS (others)')
    ax_xy.plot(eskf_pos_aligned[:, 0], eskf_pos_aligned[:, 1], '-', linewidth=1.2, label='ESKF (aligned)')
    ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.set_xlabel('East (m)')
    ax_xy.set_ylabel('North (m)')
    ax_xy.grid(True, linestyle='--', alpha=0.4)
    ax_xy.set_title('Trajectory comparison (ENU)')
    ax_xy.legend()

    ax_err = fig.add_subplot(gs[1, 0])
    ax_err.plot(t_eval, diff[:, 0], label='E error')
    ax_err.plot(t_eval, diff[:, 1], label='N error')
    ax_err.axhline(0.0, color='k', linewidth=0.8)
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Horizontal error (m)')
    ax_err.grid(True, linestyle='--', alpha=0.4)
    ax_err.legend()

    ax_vertical = fig.add_subplot(gs[1, 1])
    ax_vertical.plot(t_eval, gnss_eval[:, 2], label='GNSS U')
    ax_vertical.plot(t_eval, eskf_eval[:, 2], label='ESKF U (aligned)')
    ax_vertical.plot(t_eval, z_error, label='U error', linestyle='--')
    ax_vertical.axhline(0.0, color='k', linewidth=0.8)
    ax_vertical.set_xlabel('Time (s)')
    ax_vertical.set_ylabel('Up / error (m)')
    ax_vertical.grid(True, linestyle='--', alpha=0.4)
    ax_vertical.legend()

    text = (f"Alignment window: [{align_t0:.2f}, {align_t1:.2f}] s\n"
            f"Motion thresholds start (GNSS={gnss_motion_start:.2f}s, ESKF={eskf_motion_start:.2f}s)\n"
            f"RMSE (XY) = {rmse_xy:.3f} m\n"
            f"RMSE (Z) = {rmse_z:.3f} m\n"
            f"RMSE (3D) = {rmse_3d:.3f} m")
    fig.text(0.02, 0.02, text, fontsize=10, family='monospace', va='bottom', ha='left')

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.suptitle('ESKF vs GNSS trajectory comparison', fontsize=16)

    if args.save:
        out_path = Path(args.save)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
