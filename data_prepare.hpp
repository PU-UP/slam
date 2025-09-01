#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

// 为Eigen类型添加yaml-cpp转换支持
namespace YAML {
    template<>
    struct convert<Eigen::Vector4d> {
        static bool decode(const Node& node, Eigen::Vector4d& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }
            rhs << node[0].as<double>(), node[1].as<double>(), 
                   node[2].as<double>(), node[3].as<double>();
            return true;
        }
    };

    template<>
    struct convert<Eigen::Matrix4d> {
        static bool decode(const Node& node, Eigen::Matrix4d& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }
            for (int i = 0; i < 4; ++i) {
                if (!node[i].IsSequence() || node[i].size() != 4) {
                    return false;
                }
                for (int j = 0; j < 4; ++j) {
                    rhs(i, j) = node[i][j].as<double>();
                }
            }
            return true;
        }
    };
}

namespace CameraType {
    const int UNKOWN = 0;
    const int MONO = 1;
    const int STEREO = 2;
};
struct CameraParams {
 public:
    CameraParams() {

        model_type= " ";
        camera_name = " ";
        scaling_ratio = 1.0;
        width = 0;
        height = 0;
        projection_parameters.setZero();
        distortion_parameters.setZero();
    }
    CameraParams(const std::string& _model_type, const std::string& _camera_name,
                   double _scaling_ratio, int _image_width, int _image_height,
                   const Eigen::Vector4d& _projection_parameters, const Eigen::Vector4d& _distortion_parameters) : 
        model_type(_model_type), camera_name(_camera_name),
        scaling_ratio(_scaling_ratio), width(_image_width),
        height(_image_height), projection_parameters(_projection_parameters),
        distortion_parameters(_distortion_parameters) {
        std::cout << "load CameraParams: " 
                        << "\n model_type    : " << model_type
                        << "\n camera_name   : " << camera_name
                        << "\n scaling_ratio : " << scaling_ratio 
                        << "\n image_width   : " << width
                        << "\n image_height  : " << height
                        << "\n projection    : " << projection_parameters.transpose()
                        << "\n distortion    : " << distortion_parameters.transpose();
        }
 public:
    std::string model_type;
    std::string camera_name;
    double scaling_ratio;
    int width;
    int height;
    //k2 k3 k4 k5
    Eigen::Vector4d distortion_parameters;
    // mu mv u0 v0
    Eigen::Vector4d projection_parameters; 
};

struct IMUParams {
 public:
    IMUParams() {
        acc_n = 0.0;
        acc_w = 0.0;
        gyr_n = 0.0;
        gyr_w = 0.0;
    }
    IMUParams(double _acc_n, double _acc_w, double _gyr_n, double _gyr_w)
        : acc_n(_acc_n), acc_w(_acc_w), gyr_n(_gyr_n), gyr_w(_gyr_w) {
            std::cout << "load IMUParams : \n acc_n: " << acc_n << "\n acc_w: " << acc_w << "\n gyr_n: " << gyr_n << "\n gyr_w: " << gyr_w;
        }
 public:
    double acc_n;
    double acc_w;
    double gyr_n;
    double gyr_w;
};

struct WheelParams {
 public:
    WheelParams() {
        wheel_velocity_noise_sigma = 0;
        wheel_gyro_noise_sigma = 0;
    }
    WheelParams(double _wheel_vectory_noise_sigma, double _wheel_gyro_noise_sigma) 
        : wheel_velocity_noise_sigma(_wheel_vectory_noise_sigma), 
          wheel_gyro_noise_sigma(_wheel_gyro_noise_sigma) {
            std::cout << "load WheelParams : " << "\n wheel_gyro_noise_sigma : " << wheel_velocity_noise_sigma << "\n wheel_gyro_noise_sigma : " << wheel_gyro_noise_sigma;
        }
 public:
    double wheel_velocity_noise_sigma;
    double wheel_gyro_noise_sigma;
};

struct ExtrinsicTransform {
 public:
    ExtrinsicTransform() {
        transform = Eigen::Matrix4d::Identity();
        R = Eigen::Matrix3d::Identity();
        t = Eigen::Vector3d::Zero();
        td = 0;
    }
    ExtrinsicTransform(Eigen::Matrix<double, 4, 4> &_transform, double _td = 0) {
        if (_transform.rows() != 4 || _transform.cols() != 4) {
            std::cout<< "Invalid transform matrix!";
            return;
        }
        transform = _transform;
        R = _transform.block<3, 3>(0, 0);
        t = _transform.block<3, 1>(0, 3);
        td = _td;
        // std::cout << "load extrinsic transform: \n" << transform << "\n td : " << td;
    }
 public:
    Eigen::Matrix<double, 4, 4> transform;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double td;
};

struct CalibrationData {
 public:
    CalibrationData() {

    }
 
    CalibrationData(std::string &_version, double &_car_id, CameraParams &_intrinsic_camera, IMUParams &_intrinsic_imu, WheelParams &_intrinsic_wheel, ExtrinsicTransform &_extrinsic_body_T_cam0, ExtrinsicTransform &_extrinsic_body_T_wheel, ExtrinsicTransform &_extrinsic_body_T_cam1)
        : version(_version), car_id(_car_id), intrinsic_camera(_intrinsic_camera), intrinsic_imu(_intrinsic_imu), intrinsic_wheel(_intrinsic_wheel), extrinsic_body_T_cam0(_extrinsic_body_T_cam0), extrinsic_body_T_wheel(_extrinsic_body_T_wheel), extrinsic_body_T_cam1(_extrinsic_body_T_cam1) {
            std::cout << "\nload extrinsic_body_T_cam0: \n" << extrinsic_body_T_cam0.transform << "\n td : " << extrinsic_body_T_cam0.td
                            << "\nload extrinsic_body_T_cam1: \n" << extrinsic_body_T_cam1.transform << "\n td : " << extrinsic_body_T_cam1.td
                            << "\nload extrinsic_body_T_wheel: \n" << extrinsic_body_T_wheel.transform << "\n td : " << extrinsic_body_T_wheel.td
                            << "\nload extrinsic_wheel_T_cam0: \n" << extrinsic_wheel_T_cam0.transform << "\n td : " << extrinsic_wheel_T_cam0.td
                            << "\nload R_body_T_cam0: \n" << extrinsic_body_T_cam0.R << "\n t : " << extrinsic_body_T_cam0.t
                            << "\nload R_body_T_cam0: \n" << extrinsic_body_T_cam1.R << "\n t : " << extrinsic_body_T_cam1.t
                            << "\nload R_body_T_wheel: \n" << extrinsic_body_T_wheel.R << "\n t : " << extrinsic_body_T_wheel.t
                            << "\nload R_wheel_T_cam0: \n" << extrinsic_wheel_T_cam0.R << "\n t : " << extrinsic_wheel_T_cam0.t.transpose()
                            <<" \nload R_wheel_T_rtk: \n" <<extrinsic_wheel_T_rtk.R<< "\n t : "<< extrinsic_wheel_T_rtk.t.transpose()
                            << "\nload WheelParams : " << "\n wheel_gyro_noise_sigma : " << intrinsic_wheel.wheel_velocity_noise_sigma
                                                       << "\n wheel_gyro_noise_sigma : " << intrinsic_wheel.wheel_gyro_noise_sigma
                            << "\nload IMUParams : \n acc_n: " << intrinsic_imu.acc_n
                                                               << "\n acc_w: " << intrinsic_imu.acc_w
                                                               << "\n gyr_n: " << intrinsic_imu.gyr_n
                                                               << "\n gyr_w: " << intrinsic_imu.gyr_w
                            << "\nload CameraParams: " 
                                << "\n model_type    : " << intrinsic_camera.model_type
                                << "\n camera_name   : " << intrinsic_camera.camera_name
                                << "\n scaling_ratio : " << intrinsic_camera.scaling_ratio 
                                << "\n image_width   : " << intrinsic_camera.width
                                << "\n image_height  : " << intrinsic_camera.height
                                << "\n projection    : " << intrinsic_camera.projection_parameters.transpose()
                                << "\n distortion    : " << intrinsic_camera.distortion_parameters.transpose();

            
    }

    void Display() {
        std::cout << "\nload extrinsic_body_T_cam0: \n" << extrinsic_body_T_cam0.transform << "\n td : " << extrinsic_body_T_cam0.td
                            << "\nload extrinsic_body_T_cam1: \n" << extrinsic_body_T_cam1.transform << "\n td : " << extrinsic_body_T_cam1.td
                            << "\nload extrinsic_body_T_wheel: \n" << extrinsic_body_T_wheel.transform << "\n td : " << extrinsic_body_T_wheel.td
                            << "\nload extrinsic_wheel_T_cam0: \n" << extrinsic_wheel_T_cam0.transform << "\n td : " << extrinsic_wheel_T_cam0.td
                            << "\nload WheelParams : " << "\n wheel_gyro_noise_sigma : " << intrinsic_wheel.wheel_velocity_noise_sigma
                                                       << "\n wheel_gyro_noise_sigma : " << intrinsic_wheel.wheel_gyro_noise_sigma
                            << "\nload R_body_T_cam0: \n" << extrinsic_body_T_cam0.R << "\n t : " << extrinsic_body_T_cam0.t.transpose()
                            << "\nload R_body_T_cam0: \n" << extrinsic_body_T_cam1.R << "\n t : " << extrinsic_body_T_cam1.t.transpose()
                            << "\nload R_body_T_wheel: \n" << extrinsic_body_T_wheel.R << "\n t : " << extrinsic_body_T_wheel.t.transpose()
                            << "\nload R_wheel_T_cam0: \n" << extrinsic_wheel_T_cam0.R << "\n t : " << extrinsic_wheel_T_cam0.t.transpose()
                            <<" \nload R_wheel_T_rtk: \n" <<extrinsic_wheel_T_rtk.R<< "\n t : "<< extrinsic_wheel_T_rtk.t.transpose()
                            << "\nload IMUParams : \n acc_n: " << intrinsic_imu.acc_n
                                                               << "\n acc_w: " << intrinsic_imu.acc_w
                                                               << "\n gyr_n: " << intrinsic_imu.gyr_n
                                                               << "\n gyr_w: " << intrinsic_imu.gyr_w
                            << "\nload CameraParams: " 
                                << "\n model_type    : " << intrinsic_camera.model_type
                                << "\n camera_name   : " << intrinsic_camera.camera_name
                                << "\n scaling_ratio : " << intrinsic_camera.scaling_ratio 
                                << "\n image_width   : " << intrinsic_camera.width
                                << "\n image_height  : " << intrinsic_camera.height
                                << "\n projection    : " << intrinsic_camera.projection_parameters.transpose()
                                << "\n distortion    : " << intrinsic_camera.distortion_parameters.transpose();
    } 

    void getHeight(int& height) {
        height = intrinsic_camera.height;
    }

    void DisplayHeight() {
        std::cout  << "height : " << intrinsic_camera.height;
    }

 public:
    std::string version;
    double car_id;
    CameraParams intrinsic_camera;
    CameraParams right_intrinsic_camera;
    int camera_type = CameraType::UNKOWN;
    IMUParams intrinsic_imu;
    WheelParams intrinsic_wheel;
    double base_line;
    ExtrinsicTransform extrinsic_body_T_cam0;
    ExtrinsicTransform extrinsic_body_T_cam1;
    ExtrinsicTransform extrinsic_body_T_wheel;
    ExtrinsicTransform extrinsic_wheel_T_cam0;
    ExtrinsicTransform extrinsic_wheel_T_rtk;
};


struct RawImageData {
    cv::Mat image;
    int id;
    double timestamp;
    Eigen::Matrix4d wheel_pose;

    RawImageData() : id(-1), timestamp(0.0), wheel_pose(Eigen::Matrix4d::Identity()) {}

    RawImageData(const cv::Mat& img, int _id, double ts, const Eigen::Matrix4d& pose)
        : image(img), id(_id), timestamp(ts), wheel_pose(pose) {}
};

void LoadRawImageData(int target_id, int query_id, int length,
                      RawImageData& target_image, std::vector<RawImageData>& query_images);



// 配置文件路径 - 智能路径查找
inline std::string GetConfigPath() {
    // 尝试相对路径
    std::string relative_path = "calibration_config.yaml";
    if (std::filesystem::exists(relative_path)) {
        return relative_path;
    }
    
    // 尝试上级目录
    std::string parent_path = "../calibration_config.yaml";
    if (std::filesystem::exists(parent_path)) {
        return parent_path;
    }
    
    // 使用绝对路径作为后备
    return "/home/watermango/github/slam/calibration_config.yaml";
}

bool LoadCalibrationConfiguration(const std::string& config_path, CalibrationData& calibration_data);
