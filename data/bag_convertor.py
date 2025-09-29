#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, rosbag

GPS_TYPE  = "sensor_msgs/NavSatFix"
IMU_TYPE  = "sensor_msgs/Imu"
ODOM_TYPES = {"geometry_msgs/Twist", "geometry_msgs/TwistStamped"}

def get_first_msg_type(bag, topic):
    for _, msg, _ in bag.read_messages(topics=[topic]):
        return getattr(msg, "_type", None)
    return None

def validate_topic_category(bag, topic, category):
    tti = bag.get_type_and_topic_info()
    topics_info = getattr(tti, "topics", None) or (tti[1] if isinstance(tti, tuple) else {})
    if topic not in topics_info:
        raise ValueError(f"bag 中找不到话题: {topic}")

    t = get_first_msg_type(bag, topic)
    if t is None:
        raise ValueError(f"话题 {topic} 没有消息，无法导出")

    if category == "gps" and t != GPS_TYPE:
        raise TypeError(f"{topic} 类型应为 {GPS_TYPE}，实际为 {t}")
    if category == "imu" and t != IMU_TYPE:
        raise TypeError(f"{topic} 类型应为 {IMU_TYPE}，实际为 {t}")
    if category == "odom" and t not in ODOM_TYPES:
        raise TypeError(f"{topic} 类型应为 {', '.join(sorted(ODOM_TYPES))}，实际为 {t}")

def default_out_path(bagfile, out_dir, gps_out, imu_out, odom_out):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(bagfile)) or "."
    gps_out  = gps_out  or "gnss_data.txt"
    imu_out  = imu_out  or "bmi_imu_data.txt"
    odom_out = odom_out or "odom_data.txt"
    return (os.path.join(out_dir, gps_out),
            os.path.join(out_dir, imu_out),
            os.path.join(out_dir, odom_out))

def open_writable(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w")

def export_bag(args):
    bagfile = args.bag
    gps_topic, imu_topic, odom_topic = args.gps, args.imu, args.odom
    gps_path, imu_path, odom_path = default_out_path(
        bagfile, args.out_dir, args.gps_out, args.imu_out, args.odom_out
    )

    with rosbag.Bag(bagfile, "r") as bag:
        validate_topic_category(bag, gps_topic,  "gps")
        validate_topic_category(bag, imu_topic,  "imu")
        validate_topic_category(bag, odom_topic, "odom")

        # -------- 进度准备 --------
        tti = bag.get_type_and_topic_info()
        topics_info = getattr(tti, "topics", None) or (tti[1] if isinstance(tti, tuple) else {})
        total_msgs = sum(topics_info[t].message_count
                         for t in [gps_topic, imu_topic, odom_topic]
                         if t in topics_info)
        step = max(args.progress_step,
                   total_msgs // 100 if total_msgs >= 100 else 1)  # 默认每1%或progress_step取大
        print(f"共需处理 {total_msgs} 条消息。每 {step} 条更新一次进度。")

        count = 0
        with open_writable(gps_path) as fgps, \
             open_writable(imu_path) as fimu, \
             open_writable(odom_path) as fodom:

            for topic, msg, t in bag.read_messages(topics=[gps_topic, imu_topic, odom_topic]):
                ts = t.to_sec()

                if topic == gps_topic:
                    line = "{:.9f} {} {:.9f} {:.9f} {:.9f} ".format(
                        ts, msg.latitude, msg.longitude, msg.altitude, msg.status.status
                    )
                    line += " ".join("{:.9f}".format(c) for c in msg.position_covariance)
                    fgps.write(line + "\n")

                elif topic == imu_topic:
                    la = msg.linear_acceleration
                    av = msg.angular_velocity
                    line = "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}".format(
                        ts, la.x, la.y, la.z, av.x, av.y, av.z
                    )
                    fimu.write(line + "\n")

                elif topic == odom_topic:
                    tw = msg.twist if hasattr(msg, "twist") else msg
                    line = "{:.9f} {:.9f} {:.9f}".format(ts, tw.linear.x, tw.angular.z)
                    fodom.write(line + "\n")

                count += 1
                if count % step == 0 or count == total_msgs:
                    percent = (count / total_msgs) * 100 if total_msgs else 100
                    print(f"\r进度: {count}/{total_msgs} ({percent:.1f}%)", end="")
            print()  # 换行

    print("导出完成：")
    print("  GPS  ->", gps_path)
    print("  IMU  ->", imu_path)
    print("  ODOM ->", odom_path)

def build_parser():
    p = argparse.ArgumentParser(
        description="从 ROS bag 快速导出 GPS/IMU/Odom 文本，并带进度提示。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("bag", help="输入 .bag 文件路径")
    p.add_argument("--gps",  default="/GPS_fix", help="GPS 话题")
    p.add_argument("--imu",  default="/imu0",   help="IMU 话题")
    p.add_argument("--odom", default="/odom0",  help="Odom 话题")
    p.add_argument("--out-dir",  default=None, help="输出目录（默认与 bag 同目录）")
    p.add_argument("--gps-out",  default=None, help="GPS 输出文件名（默认 gnss_data.txt）")
    p.add_argument("--imu-out",  default=None, help="IMU 输出文件名（默认 bmi_imu_data.txt）")
    p.add_argument("--odom-out", default=None, help="ODOM 输出文件名（默认 odom_data.txt）")
    p.add_argument("--progress-step", type=int, default=100,
                   help="每处理多少条消息刷新一次进度（默认100；实际会取此值与总数1%中的较大值）")
    return p

def main():
    args = build_parser().parse_args()
    export_bag(args)

if __name__ == "__main__":
    main()
