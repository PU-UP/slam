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
        height(_image_height), distortion_parameters(_distortion_parameters),
        projection_parameters(_projection_parameters) {
        
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
        : version(_version), car_id(_car_id), intrinsic_camera(_intrinsic_camera), intrinsic_imu(_intrinsic_imu), intrinsic_wheel(_intrinsic_wheel), extrinsic_body_T_cam0(_extrinsic_body_T_cam0), extrinsic_body_T_cam1(_extrinsic_body_T_cam1), extrinsic_body_T_wheel(_extrinsic_body_T_wheel) {
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

// Configuration structure for main application
struct MainConfig {
    // Data loading parameters
    int target_image_id = 10;
    int query_image_start_id = 1;
    int query_image_count = 10;
    int query_image_length = 10;
    
    // Debug options
    bool enable_debug = true;
    bool save_intermediate = true;
    std::string output_dir = "../build/output";
    bool dr_debug = false;  // DR specific debug flag
    
    // File paths
    std::string calibration_config_path = "../calibration_config.yaml";
    std::string output_prefix = "slam_";
    
    // Data file paths
    std::string gnss_file = "../data2/gnss_data.txt";
    std::string imu_file = "../data2/bmi_imu_data.txt";
    std::string odom_file = "../data2/odom_data.txt";
};

// Load main configuration from YAML file
bool LoadMainConfiguration(const std::string& config_path, MainConfig& config);


#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace bagio {

// ---------- 共同数据基类 ----------
struct Data {
    enum class Type { IMU, ODOM, GNSS };
    explicit Data(Type t, double ts) : type(t), timestamp(ts) {}
    virtual ~Data() = default;

    Type   type;
    double timestamp; // seconds (ROS style: sec + nsec*1e-9)
};

// ---------- IMU ----------
struct ImuData : public Data {
    // File format (per your Python tool):
    // ts  linAcc.x linAcc.y linAcc.z  angVel.x angVel.y angVel.z
    ImuData() : Data(Type::IMU, 0.0) {}
    double ax{0}, ay{0}, az{0};
    double gx{0}, gy{0}, gz{0};
};

// ---------- ODOM ----------
struct OdomData : public Data {
    // File format:
    // ts  twist.linear.x  twist.angular.z
    OdomData() : Data(Type::ODOM, 0.0) {}
    double vx{0};   // linear.x
    double wz{0};   // angular.z
};

// ---------- GNSS ----------
struct GnssData : public Data {
    // NEW File format (your adjusted order):
    // ts  latitude  longitude  altitude  status  cov[0] ... cov[8]
    GnssData() : Data(Type::GNSS, 0.0) {}
    double lat{0}, lon{0}, alt{0};
    int    status{0};
    double cov[9]{};
};

// -------------- 工具：安全字符串转数 ----------------
inline bool to_double(const std::string& s, double& out) {
    char* end=nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}
inline bool to_int(const std::string& s, int& out) {
    char* end=nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

// -------------- 加载器类 ----------------
class TxtDataLoader {
public:
    struct Options {
        // 允许缺省（三个文件里任意一个可以为空字符串，表示不加载）
        std::string imu_file;
        std::string odom_file;
        std::string gnss_file;

        // 默认加载全部时间；可设置为 [t_start, t_end]（闭区间）
        double t_start = -std::numeric_limits<double>::infinity();
        double t_end   =  std::numeric_limits<double>::infinity();

        // 是否在读取时严格校验/报错
        bool strict = true;
    };

    explicit TxtDataLoader(const Options& opt) : opt_(opt) {}

    // 执行加载（可多次调用；会清空历史数据再加载）
    void load() {
        clearAll();

        if (!opt_.imu_file.empty())  loadImu(opt_.imu_file);
        if (!opt_.odom_file.empty()) loadOdom(opt_.odom_file);
        if (!opt_.gnss_file.empty()) loadGnss(opt_.gnss_file);

        // 统计时域
        computeTimeRange();

        // 合并为 unified 按时间排序（稳定）
        buildUnified();
    }

    // 1) 获取合并后的队列（深拷贝为队列给用户）
    std::queue<std::shared_ptr<Data>> unifiedQueue() const {
        std::queue<std::shared_ptr<Data>> q;
        for (auto& p : unified_) q.push(p);
        return q;
    }

    // 2) 获取分队列（2/3个，根据是否加载）
    std::queue<std::shared_ptr<ImuData>> imuQueue() const {
        std::queue<std::shared_ptr<ImuData>> q;
        for (auto& p : imu_) q.push(p);
        return q;
    }
    std::queue<std::shared_ptr<OdomData>> odomQueue() const {
        std::queue<std::shared_ptr<OdomData>> q;
        for (auto& p : odom_) q.push(p);
        return q;
    }
    std::queue<std::shared_ptr<GnssData>> gnssQueue() const {
        std::queue<std::shared_ptr<GnssData>> q;
        for (auto& p : gnss_) q.push(p);
        return q;
    }

    // 3) 时域信息
    bool hasData() const { return count_ > 0; }
    double startTime() const { return t_start_loaded_; }
    double endTime()   const { return t_end_loaded_; }
    double duration()  const { return hasData() ? (t_end_loaded_ - t_start_loaded_) : 0.0; }

    // 4) 修改时间窗口并重建（不重复读文件，直接过滤已有数据）
    void setTimeWindow(double t_start, double t_end) {
        opt_.t_start = t_start;
        opt_.t_end   = t_end;
        filterByTimeWindow();
        computeTimeRange();
        buildUnified();
    }

    // 计数
    size_t count()     const { return count_; }
    size_t imuCount()  const { return imu_.size(); }
    size_t odomCount() const { return odom_.size(); }
    size_t gnssCount() const { return gnss_.size(); }

private:
    Options opt_;

    std::vector<std::shared_ptr<ImuData>>  imu_;
    std::vector<std::shared_ptr<OdomData>> odom_;
    std::vector<std::shared_ptr<GnssData>> gnss_;
    std::vector<std::shared_ptr<Data>>     unified_;

    size_t count_{0};
    double t_start_loaded_{std::numeric_limits<double>::infinity()};
    double t_end_loaded_{-std::numeric_limits<double>::infinity()};

private:
    // ------------ 读取各文件 ------------
    void loadImu(const std::string& path) {
        std::ifstream fin(path);
        if (!fin) throw std::runtime_error("无法打开 IMU 文件: " + path);

        std::string line;
        size_t ln = 0;
        while (std::getline(fin, line)) {
            ++ln;
            if (line.empty()) continue;

            std::istringstream ss(line);
            std::vector<std::string> tok;
            std::string s;
            while (ss >> s) tok.push_back(s);
            if (tok.size() != 7) {
                if (opt_.strict)
                    throw std::runtime_error("IMU 行列数错误 @line " + std::to_string(ln));
                else
                    continue;
            }

            double ts, ax, ay, az, gx, gy, gz;
            if (!to_double(tok[0], ts) || !to_double(tok[1], ax) || !to_double(tok[2], ay) ||
                !to_double(tok[3], az) || !to_double(tok[4], gx) || !to_double(tok[5], gy) ||
                !to_double(tok[6], gz)) {
                if (opt_.strict)
                    throw std::runtime_error("IMU 数值解析失败 @line " + std::to_string(ln));
                else
                    continue;
            }

            if (ts < opt_.t_start || ts > opt_.t_end) continue;

            auto p = std::make_shared<ImuData>();
            p->timestamp = ts;
            p->ax = ax; p->ay = ay; p->az = az;
            p->gx = gx; p->gy = gy; p->gz = gz;
            imu_.push_back(std::move(p));
        }
    }

    void loadOdom(const std::string& path) {
        std::ifstream fin(path);
        if (!fin) throw std::runtime_error("无法打开 ODOM 文件: " + path);

        std::string line; size_t ln = 0;
        while (std::getline(fin, line)) {
            ++ln; if (line.empty()) continue;

            std::istringstream ss(line);
            std::vector<std::string> tok; std::string s;
            while (ss >> s) tok.push_back(s);
            if (tok.size() != 3) {
                if (opt_.strict)
                    throw std::runtime_error("ODOM 行列数错误 @line " + std::to_string(ln));
                else
                    continue;
            }

            double ts, vx, wz;
            if (!to_double(tok[0], ts) || !to_double(tok[1], vx) || !to_double(tok[2], wz)) {
                if (opt_.strict)
                    throw std::runtime_error("ODOM 数值解析失败 @line " + std::to_string(ln));
                else
                    continue;
            }

            if (ts < opt_.t_start || ts > opt_.t_end) continue;

            auto p = std::make_shared<OdomData>();
            p->timestamp = ts; p->vx = vx; p->wz = wz;
            odom_.push_back(std::move(p));
        }
    }

    void loadGnss(const std::string& path) {
        std::ifstream fin(path);
        if (!fin) throw std::runtime_error("无法打开 GNSS 文件: " + path);
    
        std::string line; size_t ln = 0;
        while (std::getline(fin, line)) {
            ++ln; if (line.empty()) continue;
    
            std::istringstream ss(line);
            std::vector<std::string> tok; std::string s;
            while (ss >> s) tok.push_back(s);
    
            if (tok.size() != 14) {
                if (opt_.strict) throw std::runtime_error("GNSS 列数应为14 @line " + std::to_string(ln));
                else continue;
            }
    
            // 尝试“新顺序”：ts lat lon alt status cov[9]
            double ts, lat, lon, alt; int status;
            bool parsed = false;
    
            auto parse_new = [&]() -> bool {
                return  to_double(tok[0], ts) && to_double(tok[1], lat) &&
                        to_double(tok[2], lon) && to_double(tok[3], alt) &&
                        to_int(tok[4], status);
            };
            auto parse_old = [&]() -> bool {
                // 旧顺序：ts status lat lon alt cov[9]
                return  to_double(tok[0], ts) && to_int(tok[1], status) &&
                        to_double(tok[2], lat) && to_double(tok[3], lon) &&
                        to_double(tok[4], alt);
            };
    
            if (parse_new()) parsed = true;
            else if (parse_old()) parsed = true;
    
            if (!parsed) {
                if (opt_.strict) throw std::runtime_error("GNSS 基本字段解析失败 @line " + std::to_string(ln));
                else continue;
            }
    
            if (ts < opt_.t_start || ts > opt_.t_end) continue;
    
            auto p = std::make_shared<GnssData>();
            p->timestamp = ts; p->lat = lat; p->lon = lon; p->alt = alt; p->status = status;
    
            bool ok = true;
            // 协方差起始列：新=5，旧=5（两种布局都是前5列放完 ts/状态/lla）
            for (int i = 0; i < 9; ++i) {
                double v;
                if (!to_double(tok[5 + i], v)) { ok = false; break; }
                p->cov[i] = v;
            }
            if (!ok) {
                if (opt_.strict) throw std::runtime_error("GNSS 协方差解析失败 @line " + std::to_string(ln));
                else continue;
            }
            gnss_.push_back(std::move(p));
        }
    }
    

    // ------------ 构建合并/统计 ------------
    void computeTimeRange() {
        count_ = imu_.size() + odom_.size() + gnss_.size();
        t_start_loaded_ = std::numeric_limits<double>::infinity();
        t_end_loaded_   = -std::numeric_limits<double>::infinity();

        auto upd = [&](double ts){
            if (ts < t_start_loaded_) t_start_loaded_ = ts;
            if (ts > t_end_loaded_)   t_end_loaded_   = ts;
        };
        for (auto& p : imu_)  upd(p->timestamp);
        for (auto& p : odom_) upd(p->timestamp);
        for (auto& p : gnss_) upd(p->timestamp);
        if (!hasData()) {
            t_start_loaded_ = 0.0; t_end_loaded_ = 0.0;
        }
    }

    void buildUnified() {
        unified_.clear();
        unified_.reserve(count_);
        for (auto& p : imu_)  unified_.push_back(p);
        for (auto& p : odom_) unified_.push_back(p);
        for (auto& p : gnss_) unified_.push_back(p);
        std::stable_sort(unified_.begin(), unified_.end(),
                         [](const std::shared_ptr<Data>& a,
                            const std::shared_ptr<Data>& b){
                             return a->timestamp < b->timestamp;
                         });
    }

    void filterByTimeWindow() {
        auto keep = [&](auto& vec){
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                        [&](const auto& p){
                            return (p->timestamp < opt_.t_start || p->timestamp > opt_.t_end);
                        }), vec.end());
        };
        keep(imu_); keep(odom_); keep(gnss_);
    }

    void clearAll() {
        imu_.clear(); odom_.clear(); gnss_.clear(); unified_.clear();
        count_ = 0; t_start_loaded_ = 0; t_end_loaded_ = 0;
    }
};

} // namespace bagio

struct Event {
    enum Type { IMU, ODOM } type;
    double t;
    std::shared_ptr<bagio::ImuData> imu;
    std::shared_ptr<bagio::OdomData> odom;
};
struct CmpEvent {
    bool operator()(const Event& a, const Event& b) const { return a.t > b.t; }
};