/**
****************************************************************************************

 * @CopyRight: 2020-2030, Positec Tech. CO.,LTD. All Rights Reserved.
 * @FilePath: dr_odo_flow.cpp
 * @Author: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Date: 2024-03-01 09:52:00
 * @Version: 0.1
 * @LastEditTime: 2024-03-01 09:52:02
 * @LastEditors: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
 * @Description: 

****************************************************************************************
*/
#include "dr/dr_odo_flow.hpp"

#include <ctime>

namespace dr_odom {

using namespace std;

DrOdoFlow::DrOdoFlow(const std::string &configure_path, CalibrationData calibration_data, bool debug_enabled) 
    : Two_(Eigen::Matrix4d::Identity()) {
    Toi_ = calibration_data.extrinsic_body_T_wheel.transform.inverse();
    
    // Load configuration
    config_ = Config::fromYaml(YAML::LoadFile(configure_path));
    config_.debug_enabled = debug_enabled;
    
    debugPrint("DR read extrinsic imu-wheel Toi: \n" + std::to_string(Toi_(0,0)) + " " + std::to_string(Toi_(0,1)) + " " + std::to_string(Toi_(0,2)) + " " + std::to_string(Toi_(0,3)) + "\n" +
               std::to_string(Toi_(1,0)) + " " + std::to_string(Toi_(1,1)) + " " + std::to_string(Toi_(1,2)) + " " + std::to_string(Toi_(1,3)) + "\n" +
               std::to_string(Toi_(2,0)) + " " + std::to_string(Toi_(2,1)) + " " + std::to_string(Toi_(2,2)) + " " + std::to_string(Toi_(2,3)) + "\n" +
               std::to_string(Toi_(3,0)) + " " + std::to_string(Toi_(3,1)) + " " + std::to_string(Toi_(3,2)) + " " + std::to_string(Toi_(3,3)));
    
    // Update parameters based on config
    mKp = config_.kp;
    mKi = config_.ki;
}


void DrOdoFlow::readConfigParameters(const std::string &config_path)
{
    YAML::Node config = YAML::LoadFile(config_path);
    std::cout << "DR configuration loaded from: " << config_path << std::endl;
}

void DrOdoFlow::Run() {
    // check if reset current node
    if (reset_flag_) {
        Reset(); // clear state
        debugPrint("DR is reseting");
        reset_flag_ = false;
    } else {
        // handle sensor data
        processWheel();
    }
}

//  清空所有状态
void DrOdoFlow::Reset() { 
    debugPrint("Dr reset");
    Two_ = Eigen::Matrix4d::Identity();

    m_q0 = 1.0, m_q1 = 0.0, m_q2 = 0.0, m_q3 = 0.0;
    mIntegralFBx = 0.0, mIntegralFBy = 0.0, mIntegralFBz = 0.0;

    static_gyro_sum_ = Eigen::Vector3d(0., 0., 0.);
    static_gyro_count_ = 0;
    {
        // std::lock_guard<std::mutex> lock(bias_mutex_);
        static_bias_acc_ = Eigen::Vector3d(0., 0., 0.);
        static_bias_gyro_ = Eigen::Vector3d(0., 0., 0.);
    }

    last_imu_time_ = 0.;
    last_rqy_ = Eigen::Vector3d(0., 0., 0.);
    last_R_ = Eigen::Matrix3d::Identity();
    last_P_ = Eigen::Vector3d(0., 0., 0.);

    // imu_buffer_.clear();
    // wheel_odom_buffer_.clear();

    current_x_ = 0.;
    current_y_ = 0.;
    last_wheel_time_ = 0.;
    slip_flag_ = false;
    last_slip_flag_ = false;

}

Eigen::Matrix3d DrOdoFlow::eulerAnglesToRotationMatrix(Eigen::Vector3d &theta) {
    Eigen::Matrix3d R_x;    // 计算旋转矩阵的X分量
    R_x <<  1, 0, 0,
            0, cos(theta[0]), -sin(theta[0]),
            0, sin(theta[0]), cos(theta[0]);

    Eigen::Matrix3d R_y;    // 计算旋转矩阵的Y分量
    R_y <<  cos(theta[1]), 0, sin(theta[1]),
            0, 1, 0,
            -sin(theta[1]), 0, cos(theta[1]);

    Eigen::Matrix3d R_z;    // 计算旋转矩阵的Z分量
    R_z <<  cos(theta[2]), -sin(theta[2]), 0,
            sin(theta[2]), cos(theta[2]), 0,
            0, 0, 1;
    Eigen::Matrix3d R = R_z * R_y * R_x;
    return R;
}

// void DrOdoFlow::getPose(double &time, Eigen::Matrix<double, 6, 1> &pose) {
//     time = last_wheel_time_;
//     pose(0) = current_x_;
//     pose(1) = current_y_;
//     pose(2) = 0;
//     pose(3) = last_rqy_(0);
//     pose(4) = last_rqy_(1);
//     pose(5) = last_rqy_(2);
// }


void DrOdoFlow::setImuBias(const Eigen::Vector3d &Ba_static, const Eigen::Vector3d &Bg_static)
{
    if(!Bg_static.isZero())
    {
        // std::lock_guard<std::mutex> lock(bias_mutex_);

			if ( (static_bias_acc_ - Ba_static).squaredNorm() > 1e-6
				|| (static_bias_gyro_ - Bg_static).squaredNorm() > 1e-6){
				std::cout << " static_bias_acc_: " << static_bias_acc_.transpose() << " norm: " << static_bias_acc_.norm() << std::endl;
				std::cout << " static_bias_gyro_: " << static_bias_gyro_.transpose() << " norm: " << static_bias_gyro_.norm() << std::endl;
			}

			static_bias_acc_ = Ba_static;
			static_bias_gyro_ = Bg_static;
    }
}

void DrOdoFlow::setImu(double current_time, double ax, double ay, double az, double gx, double gy, double gz) {
    Eigen::Matrix<double, 6, 1> imu;
    imu << ax, ay, az, gx, gy, gz;
    imu_buffer_.emplace_back(current_time, imu);
        // 超过配置的最大大小
    while (imu_buffer_.size() > config_.buffer_max_size_imu) {
        debugPrint(std::to_string(current_time) + " imu_buffer_ size too big, over " + std::to_string(config_.buffer_max_size_imu) + ", need to pop front: " + std::to_string(imu_buffer_.front().first));
        imu_buffer_.pop_front();
    }
}

void DrOdoFlow::setWheel(double current_time, double vel_linear, double vel_angular) {
    wheel_odom_buffer_.emplace_back(current_time, Eigen::Vector2d(vel_linear, vel_angular));
    // 超过配置的最大大小
    while (wheel_odom_buffer_.size() > config_.buffer_max_size_wheel) {
        debugPrint(std::to_string(current_time) + " wheel_odom_buffer_ size too big, over " + std::to_string(config_.buffer_max_size_wheel) + ", need to pop front: " + std::to_string(wheel_odom_buffer_.front().first));
        wheel_odom_buffer_.pop_front();
    }
}

void DrOdoFlow::setSlipFlag(bool slip_flag) {
    slip_flag_ = slip_flag;
}

//set acc and gyro
void DrOdoFlow::processImu(double current_time, const Eigen::Matrix<double, 6, 1> &imu, bool is_static) {

    // SLAM_LOG_INFO() << std::to_string(current_time) << " static_bias_acc_: " << static_bias_acc_.transpose();
    // SLAM_LOG_INFO() << std::to_string(current_time) << " static_bias_gyro_: " << static_bias_gyro_.transpose();
    Eigen::Vector3d acc = imu.head(3);
    Eigen::Vector3d gyro = imu.tail(3); // * ratio,

    {
        // std::lock_guard<std::mutex> lock(bias_mutex_);
			#if 0
        if (is_static && (fabs(gyro(0)) < 0.015) && (fabs(gyro(1)) < 0.015) && (fabs(gyro(2)) < 0.015) &&
            (static_gyro_count_ < 1000)) {

            calculateAccBias(acc);
            calculateGyroBias(gyro);

            if(static_gyro_count_ % 100 == 0) {
                std::cout << std::to_string(current_time) 
                                << " static_gyro_count_: " << static_gyro_count_
                                << " static acc bias: " << static_bias_acc_.transpose() << " norm: " << static_bias_acc_.norm()
                                << " static gyro bias: " << static_bias_gyro_.transpose() << " norm: " << static_bias_gyro_.norm() << std::endl;
            }
        }
			#endif

        acc = acc - static_bias_acc_;
        gyro = gyro - static_bias_gyro_;

        // if (is_static && (fabs(gyro(0)) < 0.015) && (fabs(gyro(1)) < 0.015) && (fabs(gyro(2)) < 0.015)) {
        if (is_static) {
            if((fabs(gyro(0)) < config_.static_gyro_threshold) && (fabs(gyro(1)) < config_.static_gyro_threshold) && (fabs(gyro(2)) < config_.static_gyro_threshold))
            {
                // gyro = Eigen::Vector3d(0., 0., 0.);
            }
            else
            {
                debugPrint(std::to_string(current_time) + " gyro data when static: " + std::to_string(gyro(0)) + " " + std::to_string(gyro(1)) + " " + std::to_string(gyro(2)) + " norm: " + std::to_string(gyro.norm()));
            }
        }
    }

    // 利用外参修正imu读数（转到wheel_odom坐标系）
    gyro = (Toi_.block<3, 3>(0, 0) * gyro).eval();
    acc = (Toi_.block<3, 3>(0, 0) * acc).eval();

    double delta_time = current_time - last_imu_time_;

    // if (last_imu_time_ > 0.01 && (current_time - last_imu_time_ < 0.2)) {
    if (last_imu_time_ > 0.01 && (delta_time > 0.0) && (delta_time < 1.0)) {
        if (delta_time >= config_.time_diff_warning)
        {
            debugPrint("IMU time diff too big " + std::to_string(delta_time) + "s, current time: " + std::to_string(current_time) + ", last_imu_time_: " + std::to_string(last_imu_time_));
        }
        // 静止不做滤波，避免角度漂移
        if(!is_static)
        {					
			      last_rqy_ = updateIMU(gyro(0), gyro(1), gyro(2), acc(0), acc(1), acc(2), delta_time);
        }


    } else {
        if(delta_time >= config_.time_diff_error)
        {
            debugPrint("IMU time diff too big " + std::to_string(delta_time) + "s, current time: " + std::to_string(current_time) + ", last_imu_time_: " + std::to_string(last_imu_time_));
        }
        else if(delta_time <= 0.0)
        {
            debugPrint("IMU time diff smaller than 0, " + std::to_string(delta_time) + "s, current time: " + std::to_string(current_time) + ", last_imu_time_: " + std::to_string(last_imu_time_));
        }
        else
        {
            debugPrint(" last_imu_time_ abnormal: " + std::to_string(last_imu_time_));
        }
    }

    last_imu_time_ = current_time;
}


void DrOdoFlow::processWheel() {
    
    while (!wheel_odom_buffer_.empty()) {

        // SLAM_LOG_INFO() << " imu_buffer_ size: " << imu_buffer_.size() 
        //                 << " wheel_odom_buffer_ size: " << wheel_odom_buffer_.size() 
        //                 << " wheel front: " << std::to_string(wheel_odom_buffer_.front().first)
        //                 << " wheel back : " << std::to_string(wheel_odom_buffer_.back().first)
        //                 << " imu front: " << std::to_string(imu_buffer_.front().first)
        //                 << " imu back : " << std::to_string(imu_buffer_.back().first);

        double current_time = wheel_odom_buffer_.front().first;
        double vel_linear = wheel_odom_buffer_.front().second(0);
        double vel_angular = wheel_odom_buffer_.front().second(1);

        wheel_odom_buffer_.pop_front();

        // 只处理当前能够完成时间同步的轮子数据，依据：当前IMU最新一帧的时间戳大于当前轮子时间戳(改为外部component去筛选)
        // if (current_time >= imu_buffer_.back().first)
        // {
        //     SLAM_LOG_INFO() << std::to_string(current_time) << " time sync failed, not possible.";
        //     break;
        // }

        if (last_wheel_time_ > 0.1) {
            bool is_static = false;
            if ((fabs(vel_linear) < 1e-6) && (fabs(vel_angular) < 1e-6)) { //not support for big time lag
                is_static = true;
            } else {
                is_static = false;
            }
   
            while ((!imu_buffer_.empty()) && (imu_buffer_.front().first < current_time)) {
                processImu(imu_buffer_.front().first, imu_buffer_.front().second, is_static);
                imu_buffer_.pop_front();
            }

            // 二维信息
            // double travelled_distance = (current_time - last_wheel_time_) * vel_linear;
            // current_x_ += cos(last_rqy_(2)) * travelled_distance;
            // current_y_ += sin(last_rqy_(2)) * travelled_distance;
            // Two(0, 3) = current_x_;
            // Two(1, 3) = current_y_;
            // 250806打滑或者悬挂状态下，速度置零
            if (slip_flag_ ){
                vel_linear = 0;
            }
            if(last_slip_flag_ != slip_flag_) debugPrint("current slip flag: " + std::to_string(slip_flag_));
            last_slip_flag_ = slip_flag_;
						// 积累三轴的速度，计算出三轴的平移量
            Eigen::Vector3d local_vel_vec(vel_linear, 0, 0);
            double dt = current_time - last_wheel_time_;
            // 只更新时间变化小于1秒并且轮子速度小于配置的最大速度的数据
            if (dt < 0 || dt > 1 || fabs(vel_linear) > config_.velocity_max)
            {
                debugPrint(std::to_string(current_time) + " vel_linear: " + std::to_string(vel_linear) + " m/s dt: " + std::to_string(dt) + " s");
            }
            else
            {
                last_R_ = eulerAnglesToRotationMatrix(last_rqy_);
                Eigen::Vector3d world_vel_vec = last_R_ * local_vel_vec;
                last_P_ += world_vel_vec * dt;

                if ((world_vel_vec * dt).norm() > config_.displacement_max) {
                  debugPrint(std::to_string(current_time) + " world_vel_vec: " + std::to_string(world_vel_vec(0)) + " " + std::to_string(world_vel_vec(1)) + " " + std::to_string(world_vel_vec(2)) + " m/s(vec) wheel vel: " + std::to_string(vel_linear) + " m/s, dt: " + std::to_string(dt) + " s, norm: " + std::to_string((world_vel_vec * dt).norm()) + " m");
                }

                Eigen::Matrix<double, 4, 4> Two = Eigen::Matrix<double, 4, 4>::Identity();
                Two.block<3, 1>(0, 3) = last_P_;
                Two.block<3, 3>(0, 0) = last_R_;
                Two_ = Two;

            }
        } else {
            debugPrint(" Start Process Data, last_wheel_time_ smaller than 0.1, is " + std::to_string(last_wheel_time_) + ", clear all buffer.");
            imu_buffer_.clear();
            wheel_odom_buffer_.clear();
        }
        last_wheel_time_ = current_time;
    }
}
#if 0
void DrOdoFlow::calculateAccBias(const Eigen::Vector3d &acc)
{
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 9.8);

    static_acc_count_++;
    static_acc_sum_ += acc;

    Eigen::Vector3d avg_acc = static_acc_sum_ / static_acc_count_;

    Eigen::Matrix3d R0 = g2R(avg_acc);

    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Eigen::Matrix3d R_GtoI = R0.inverse();

    static_bias_acc_ = avg_acc - R_GtoI * gravity;
}


void DrOdoFlow::calculateGyroBias(const Eigen::Vector3d& gyr) {
    // std::lock_guard<std::mutex> lock(bias_mutex_);

    static_gyro_count_++;
    static_gyro_sum_ += gyr;
    static_bias_gyro_ = static_gyro_sum_ / double(static_gyro_count_);
}
#endif

// double DrOdoFlow::radiansToDegrees(double radians) {
//     return radians * (180.0 / M_PI);
// }

//by mahony
Eigen::Vector3d DrOdoFlow::updateIMU(double gx, double gy, double gz, double ax, double ay, double az, double dT) {
    
    // static Eigen::Matrix3d Rs = Eigen::Matrix3d::Identity();

    // // 获取当前姿态
    // Eigen::Quaterniond cur_q(m_q3, m_q0, m_q1, m_q2);
    // cur_q.normalize();

    // // 根据角速度计算相对量
    // Eigen::Vector3d un_gyr(gx, gy, gz);
    // Eigen::Quaterniond dq;
    // Eigen::Vector3d half_theta = un_gyr * dT;
    // half_theta /= 2.0;
    // dq.w() = 1.0;
    // dq.x() = half_theta.x();
    // dq.y() = half_theta.y();
    // dq.z() = half_theta.z();
    // dq.normalize();

    // cur_q = cur_q * dq;

    // Rs *= dq.toRotationMatrix();

    // m_q0 = cur_q.x();
    // m_q1 = cur_q.y();
    // m_q2 = cur_q.z();
    // m_q3 = cur_q.w();

    // cur_q.normalize();

    // // Eigen::Vector3d tmp_rpy = cur_q.toRotationMatrix().eulerAngles(0, 1, 2);
    // Eigen::Vector3d n = Rs.col(0);
    // Eigen::Vector3d o = Rs.col(1);
    // Eigen::Vector3d a = Rs.col(2);

    // Eigen::Vector3d ypr(3);
    // double y = atan2(n(1), n(0));
    // double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    // double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    // ypr(0) = y;
    // ypr(1) = p;
    // ypr(2) = r;

    // // 计算ypr
    // double roll = ypr.z();
    // double pitch = ypr.y();
    // double yaw = ypr.x();

    // // SLAM_LOG_HIGHLIGHT() << " yaw: " << yaw * 180 / M_PI;

    // return {roll, pitch, yaw};

    // 以下暂时不运行
    
    
    // 得到加速度计的方向向量
    double recipNorm = sqrt(ax * ax + ay * ay + az * az);
    ax /= recipNorm;
    ay /= recipNorm;
    az /= recipNorm;

    // 当前重力方向
    double halfvx = m_q1 * m_q3 - m_q0 * m_q2;
    double halfvy = m_q0 * m_q1 + m_q2 * m_q3;
    double halfvz = m_q0 * m_q0 - 0.5 + m_q3 * m_q3;

    // 加速度计方向向量和当前重力方向向量叉乘，得到误差向量
    double halfex = (ay * halfvz - az * halfvy);
    double halfey = (az * halfvx - ax * halfvz);
    double halfez = (ax * halfvy - ay * halfvx);

    Eigen::Vector3d error_vector(halfex, halfey, halfez);
    static double abnormal_error_vector_duration = 0;
    static double sum_dt = 0;
    sum_dt += dT;
    if(error_vector.norm() > 0.2)
    {
        abnormal_error_vector_duration += dT;
        debugPrint(" Robot run time: " + std::to_string(sum_dt) + " s, abnormal error vector duration: " + std::to_string(abnormal_error_vector_duration) + " s, current error_vector: " + std::to_string(error_vector(0)) + " " + std::to_string(error_vector(1)) + " " + std::to_string(error_vector(2)) + ", norm: " + std::to_string(error_vector.norm()) + " current gravity vector: " + std::to_string(halfvx) + " " + std::to_string(halfvy) + " " + std::to_string(halfvz) + " current acc vector:" + std::to_string(ax) + " " + std::to_string(ay) + " " + std::to_string(az) + " norm: " + std::to_string(recipNorm));
    }
    else
    {
        abnormal_error_vector_duration = 0;
    }

    // 积分项
    mIntegralFBx += mKi * halfex * dT;
    mIntegralFBy += mKi * halfey * dT;
    mIntegralFBz += mKi * halfez * dT;

    gx += mIntegralFBx;
    gy += mIntegralFBy;
    gz += mIntegralFBz;

    // 常量
    gx += mKp * halfex;
    gy += mKp * halfey;
    gz += mKp * halfez;

    // 基于PI得到修正后的角速度
    gx *= (0.5 * dT);
    gy *= (0.5 * dT);
    gz *= (0.5 * dT);

    // 积分更新
    const double qa = m_q0;
    const double qb = m_q1;
    const double qc = m_q2;
    m_q0 += (-qb * gx - qc * gy - m_q3 * gz);
    m_q1 += (qa * gx + qc * gz - m_q3 * gy);
    m_q2 += (qa * gy - qb * gz + m_q3 * gx);
    m_q3 += (qa * gz + qb * gy - qc * gx);

    // 归一化
    recipNorm = sqrt(m_q0 * m_q0 + m_q1 * m_q1 + m_q2 * m_q2 + m_q3 * m_q3);
    // double recipNorm = sqrt(m_q0 * m_q0 + m_q1 * m_q1 + m_q2 * m_q2 + m_q3 * m_q3);
    m_q0 /= recipNorm;
    m_q1 /= recipNorm;
    m_q2 /= recipNorm;
    m_q3 /= recipNorm;

    // // 归一化 (w,x,y,z)
    // Eigen::Quaterniond cur_q(m_q0, m_q1, m_q2, m_q3);
    // cur_q.normalize();
    // // 获取当前的旋转矩阵
    // last_R_ = cur_q.toRotationMatrix();

    // 计算ypr
    double roll = atan2(m_q0 * m_q1 + m_q2 * m_q3, 0.5 - m_q1 * m_q1 - m_q2 * m_q2);
    double pitch = asin(-2.0 * (m_q1 * m_q3 - m_q0 * m_q2));
    double yaw = atan2(m_q1 * m_q2 + m_q0 * m_q3, 0.5 - m_q2 * m_q2 - m_q3 * m_q3);

    return {roll, pitch, yaw};
}

}//namespace dr_odom

