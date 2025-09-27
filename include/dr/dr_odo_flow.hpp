/**
****************************************************************************************

 * @CopyRight: 2020-2030, Positec Tech. CO.,LTD. All Rights Reserved.
 * @FilePath: dr_odo_flow.hpp
 * @Author: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Date: 2024-03-01 10:00:34
 * @Version: 0.1
 * @LastEditTime: 2024-03-01 10:00:35
 * @LastEditors: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Description: 

****************************************************************************************
*/
#ifndef SRC_POSE_SOURCE_DR_ODO_FLOW_HPP_
#define SRC_POSE_SOURCE_DR_ODO_FLOW_HPP_

#include <mutex>
#include <chrono>
#include <thread>
#include <deque>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include "../data_prepare.hpp"


namespace dr_odom {
/**
 * @brief fusion imu and odom class
 * 
 */

static Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr)
{
    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<double, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix<double, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix<double, 3, 3> Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;

    return R0;
}


class DrOdoFlow {
public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Configuration structure for DR parameters
    struct Config {
        // Mahony filter parameters
        double kp = 1.0;
        double ki = 0.0001;
        
        // IMU processing parameters
        double static_gyro_threshold = 0.015;
        double time_diff_warning = 0.1;
        double time_diff_error = 1.0;
        size_t buffer_max_size_imu = 400;
        
        // Wheel processing parameters
        size_t buffer_max_size_wheel = 100;
        double velocity_max = 0.8;
        double displacement_max = 0.4;
        
        // Bias estimation parameters
        int acc_bias_update_samples = 100;
        int gyro_bias_update_samples = 1000;
        
        // Debug flag
        bool debug_enabled = false;
        
        // Load configuration from YAML node
        static Config fromYaml(const YAML::Node& node) {
            Config config;
            
            if (node["dr"]) {
                const auto& dr_config = node["dr"];
                
                if (dr_config["mahony"]) {
                    const auto& mahony = dr_config["mahony"];
                    if (mahony["kp"]) config.kp = mahony["kp"].as<double>();
                    if (mahony["ki"]) config.ki = mahony["ki"].as<double>();
                }
                
                if (dr_config["imu"]) {
                    const auto& imu = dr_config["imu"];
                    if (imu["static_gyro_threshold"]) config.static_gyro_threshold = imu["static_gyro_threshold"].as<double>();
                    if (imu["time_diff_warning"]) config.time_diff_warning = imu["time_diff_warning"].as<double>();
                    if (imu["time_diff_error"]) config.time_diff_error = imu["time_diff_error"].as<double>();
                    if (imu["buffer_max_size"]) config.buffer_max_size_imu = imu["buffer_max_size"].as<size_t>();
                }
                
                if (dr_config["wheel"]) {
                    const auto& wheel = dr_config["wheel"];
                    if (wheel["buffer_max_size"]) config.buffer_max_size_wheel = wheel["buffer_max_size"].as<size_t>();
                    if (wheel["velocity_max"]) config.velocity_max = wheel["velocity_max"].as<double>();
                    if (wheel["displacement_max"]) config.displacement_max = wheel["displacement_max"].as<double>();
                }
                
                if (dr_config["bias"]) {
                    const auto& bias = dr_config["bias"];
                    if (bias["acc_bias_update_samples"]) config.acc_bias_update_samples = bias["acc_bias_update_samples"].as<int>();
                    if (bias["gyro_bias_update_samples"]) config.gyro_bias_update_samples = bias["gyro_bias_update_samples"].as<int>();
                }
            }
            
            // Load debug flag from main debug section
            if (node["debug"] && node["debug"]["dr_debug"]) {
                config.debug_enabled = node["debug"]["dr_debug"].as<bool>();
            }
            
            return config;
        }
    };
    
    DrOdoFlow(const std::string &configure_path, CalibrationData calibration_data, bool debug_enabled = false);

    ~DrOdoFlow() {};

    void Run();  // handle sensor data

    void Reset(); // clear history state for reset

public:
    void readConfigParameters(const std::string &config_path);

    Eigen::Matrix3d eulerAnglesToRotationMatrix(Eigen::Vector3d &theta);

    void setImuBias(const Eigen::Vector3d &Ba_static, const Eigen::Vector3d &Bg_static);

    //set acc and gyro
    void setImu(double current_time, double ax, double ay, double az, double gx, double gy, double gz);

    void setWheel(double current_time, double vel_linear, double vel_angular);
        
    void setSlipFlag(bool slip_flag);
    //get time and pose:x,y,z,r,p,y
    // void getPose(double &time, Eigen::Matrix<double, 6, 1> &pose);

    void getPose(double &time, Eigen::Matrix4d &pose) {
        time = last_wheel_time_;
        pose = Two_;
    }

    void getPoseInIMUFrame(double &time, Eigen::Matrix4d& pose) {
        time = last_wheel_time_;
        // pose = Two_ * Toi_;
        // Eigen::Matrix4d Tio_ = Toi_.inverse();
        // SLAM_LOG_HIGHLIGHT() << "DR read extrinsic imu-wheel Toi: \n" << Toi_; 
        Eigen::Matrix3d tmp_r = Two_.block<3,3>(0,0);
        Eigen::Vector3d tmp_t = Two_.block<3,1>(0,3);

        pose.block<3,1>(0,3) = Toi_.block<3,3>(0,0).inverse() * 
            (tmp_t + tmp_r * Toi_.block<3,1>(0,3) - Toi_.block<3,1>(0,3)); 

        Eigen::Matrix3d R = Toi_.block<3,3>(0,0).inverse() * tmp_r;

        //LM-NOTE 2024.08.22
        //由于IMU装配导致机体超前时 X轴朝左 Y轴朝后 所以在转换到IMU坐标系后需要把装配旋转角度补偿回去
        double yaw = R2ypr(Toi_.block<3,3>(0,0)).x();
        R = ypr2R(Eigen::Vector3d{yaw, 0, 0}) * R;
        pose.block<3,3>(0,0) = R;
        // double tmp_x = pose(0, 3);
        // double tmp_y = pose(1, 3);
        // pose(0, 3) = tmp_y;
        // pose(1, 3) = -tmp_x;
    }


private:
    // Configuration
    Config config_;
    
    // Debug control
    void debugPrint(const std::string& message) const {
        if (config_.debug_enabled) {
            std::cout << message << std::endl;
        }
    }
    
    int reset_flag_ = false;

private:
    void processImu(double current_time, const Eigen::Matrix<double, 6, 1> &imu, bool bStatic);

    void processWheel();
#if 0
    void calculateAccBias(const Eigen::Vector3d &acc);
    void calculateGyroBias(const Eigen::Vector3d &gry);
#endif
    Eigen::Vector3d updateIMU(double gx, double gy, double gz, double ax, double ay, double az, double dT);

    // static double radiansToDegrees(double radians);

    std::mutex imu_data_mutex_;
    std::mutex wheel_data_mutex_;

    std::mutex bias_mutex_;

    double mKp = 1.0;
    double mKi = 0.0001;
    double m_q0 = 1.0, m_q1 = 0.0, m_q2 = 0.0, m_q3 = 0.0;
    double mIntegralFBx = 0.0, mIntegralFBy = 0.0, mIntegralFBz = 0.0;

    int static_acc_count_ = 0;
    Eigen::Vector3d static_acc_sum_ = Eigen::Vector3d(0., 0., 0.);
    Eigen::Vector3d static_bias_acc_ = Eigen::Vector3d(0., 0., 0.);

    int static_gyro_count_ = 0;
    Eigen::Vector3d static_gyro_sum_ = Eigen::Vector3d(0., 0., 0.);
    Eigen::Vector3d static_bias_gyro_ = Eigen::Vector3d(0., 0., 0.);

    double last_imu_time_ = 0.;
    double last_wheel_time_ = 0.;
    Eigen::Vector3d last_rqy_ = Eigen::Vector3d(0., 0., 0.);
    Eigen::Matrix3d last_R_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d last_P_ = Eigen::Vector3d(0., 0., 0.);

    std::deque<std::pair<double, Eigen::Matrix<double, 6, 1>>> imu_buffer_;
    std::deque<std::pair<double, Eigen::Vector2d>> wheel_odom_buffer_;

    // odom pose
    double current_x_ = 0.;
    double current_y_ = 0.;
    // imu与odom相对外参
    Eigen::Matrix4d Toi_ = Eigen::Matrix4d::Identity();
    // wheel_odom in world
    Eigen::Matrix4d Two_ = Eigen::Matrix4d::Identity(); 

    bool slip_flag_ = false;
    bool last_slip_flag_ = false;
    bool last_suspended_flag_ = false;
};

#endif //SRC_POSE_SOURCE_DR_ODO_FLOW_HPP_
} // dr_odom
