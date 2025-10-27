#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import evo.tools.file_interface as fio
from evo.core.trajectory import PoseTrajectory3D

def nearest_time_match(t_ref, t_est, max_diff):
    """把 t_ref 上的每个时间点在 t_est 里找最近邻（<=max_diff），返回匹配索引对。"""
    matches_ref = []
    matches_est = []
    j = 0
    n_est = len(t_est)
    for i, tr in enumerate(t_ref):
        # 双向逼近最近邻
        while j + 1 < n_est and abs(t_est[j + 1] - tr) <= abs(t_est[j] - tr):
            j += 1
        if abs(t_est[j] - tr) <= max_diff:
            matches_ref.append(i)
            matches_est.append(j)
    return np.asarray(matches_ref, dtype=int), np.asarray(matches_est, dtype=int)

def fit_rigid_xy(est_xy, ref_xy):
    """2D 刚体（无尺度）配准，返回 R(2x2), t(2,)"""
    ref_mean = ref_xy.mean(axis=0)
    est_mean = est_xy.mean(axis=0)
    A = est_xy - est_mean
    B = ref_xy - ref_mean
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = ref_mean - (R @ est_mean)
    return R, t

def apply_xy_transform(traj_est, R, t, shift_z=False, z_ref=None):
    """
    把 XY 刚体变换应用到整条估计轨迹（只旋转XY；是否平移Z可选）。
    返回一个新的 PoseTrajectory3D。
    """
    P = traj_est.positions_xyz
    P_xy = (R @ P[:, :2].T).T + t
    P_new = P.copy()
    P_new[:, 0:2] = P_xy
    if shift_z and z_ref is not None:
        # 把Z整体平移到与参考首帧一致（可选）
        dz = z_ref - P[0, 2]
        P_new[:, 2] = P[:, 2] + dz
    return PoseTrajectory3D(
        positions_xyz=P_new,
        orientations_quat_wxyz=traj_est.orientations_quat_wxyz,
        timestamps=traj_est.timestamps
    )

def align_xy_using_front_segment(traj_est, traj_ref, align_time, t_max_diff):
    # 取参考前 align_time 秒的索引
    t0 = traj_ref.timestamps[0]
    mask_ref = (traj_ref.timestamps - t0) < align_time
    idx_ref_sub = np.where(mask_ref)[0]
    if len(idx_ref_sub) < 3:
        raise RuntimeError("参考轨迹在对齐窗口内点数不足（<3）。请增大 --align_time。")

    # 在窗口内的参考时间与估计全量时间做最近邻匹配
    ref_t_sub = traj_ref.timestamps[idx_ref_sub]
    i_ref_win, i_est_win = nearest_time_match(ref_t_sub, traj_est.timestamps, t_max_diff)
    if len(i_ref_win) < 3:
        raise RuntimeError("时间匹配后的样本太少（<3）。请增大 --align_time 或 --t_max_diff。")

    ref_xy = traj_ref.positions_xyz[idx_ref_sub[i_ref_win], :2]
    est_xy = traj_est.positions_xyz[i_est_win, :2]

    R, t = fit_rigid_xy(est_xy, ref_xy)
    traj_est_aligned = apply_xy_transform(traj_est, R, t)
    return traj_est_aligned, R, t

def compute_xy_errors(traj_ref, traj_est_aligned, t_max_diff):
    """全时域最近邻匹配，计算 XY 平移误差和对应时间。"""
    i_ref, i_est = nearest_time_match(traj_ref.timestamps, traj_est_aligned.timestamps, t_max_diff)
    if len(i_ref) < 2:
        raise RuntimeError("全局误差计算的有效匹配过少。请增大 --t_max_diff。")
    ref_xy = traj_ref.positions_xyz[i_ref, :2]
    est_xy = traj_est_aligned.positions_xyz[i_est, :2]
    err = np.linalg.norm(est_xy - ref_xy, axis=1)
    t_rel = traj_ref.timestamps[i_ref] - traj_ref.timestamps[0]
    return t_rel, err

def summarize_errors(err):
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "min": float(np.min(err)),
        "std": float(np.std(err)),
        "count": int(err.size),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Align est to ref using the first --align_time seconds (XY only), save aligned.txt and show plots."
    )
    ap.add_argument("traj_est", help="Estimated trajectory (TUM format)")
    ap.add_argument("traj_ref", help="Reference trajectory (TUM format)")
    ap.add_argument("--align_time", type=float, default=10.0, help="Seconds from the start of reference used for alignment")
    ap.add_argument("--t_max_diff", type=float, default=0.05, help="Timestamp tolerance for matching (seconds)")
    ap.add_argument("--save", default="aligned.txt", help="Output aligned trajectory file")
    ap.add_argument("--save_plot", default=None, help="If set, save plots to this prefix, e.g., results (will create results_xy.png and results_err.png)")
    ap.add_argument("--shift_z", action="store_true", help="Also shift Z so that the first Z matches reference (rotation still XY-only)")
    args = ap.parse_args()

    # 读取
    traj_ref = fio.read_tum_trajectory_file(args.traj_ref)
    traj_est = fio.read_tum_trajectory_file(args.traj_est)

    # 对齐
    traj_est_aligned, R, t = align_xy_using_front_segment(
        traj_est, traj_ref, args.align_time, args.t_max_diff
    )

    # 可选：Z 整体平移到参考首帧
    if args.shift_z:
        traj_est_aligned = apply_xy_transform(
            traj_est_aligned, np.eye(2), np.zeros(2), shift_z=True, z_ref=traj_ref.positions_xyz[0, 2]
        )

    # 写文件
    fio.write_tum_trajectory_file(args.save, traj_est_aligned)
    print(f"[✅] Saved aligned trajectory to {args.save}")
    print("[ℹ️] XY rotation matrix:")
    print(R)
    print("[ℹ️] XY translation vector:", t)

    # 误差计算（全时域）
    t_rel, err = compute_xy_errors(traj_ref, traj_est_aligned, args.t_max_diff)
    stats = summarize_errors(err)
    print(
        "[📊] XY error (m): "
        f"RMSE={stats['rmse']:.3f}, mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
        f"std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}, N={stats['count']}"
    )

    # 图1：XY轨迹
    plt.figure()
    plt.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], label="Reference")
    plt.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], "--", label="Before align")
    plt.plot(traj_est_aligned.positions_xyz[:, 0], traj_est_aligned.positions_xyz[:, 1], label="After align")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.axis("equal"); plt.legend()
    plt.title(f"XY Trajectory (align first {args.align_time}s, t_max_diff={args.t_max_diff}s)")

    if args.save_plot:
        plt.savefig(f"{args.save_plot}_xy.png", dpi=200, bbox_inches="tight")

    # 图2：XY误差随时间
    plt.figure()
    plt.plot(t_rel, err)
    plt.xlabel("Time from start [s]"); plt.ylabel("XY error [m]")
    plt.title("XY translation error vs time")

    if args.save_plot:
        plt.savefig(f"{args.save_plot}_err.png", dpi=200, bbox_inches="tight")

    # 展示
    plt.show()

if __name__ == "__main__":
    main()
