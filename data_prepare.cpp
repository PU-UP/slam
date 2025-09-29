#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <filesystem>
#include "data_prepare.hpp"
#include <fstream>
#include <sstream>
#include <map>



void LoadRawImageData(int target_id, int query_id, int length,
                      RawImageData& target_image, std::vector<RawImageData>& query_images) {
  std::string image_path = "/home/watermango/data/raw_data_for_loop_closure/image/";
  std::string info = "/home/watermango/data/raw_data_for_loop_closure/keyframes_info.txt";

  // 读取关键帧信息文件
  std::ifstream info_file(info);
  if (!info_file.is_open()) {
    std::cerr << "无法打开关键帧信息文件: " << info << std::endl;
    return;
  }

  // 存储所有关键帧信息的map
  std::map<int, std::pair<double, Eigen::Matrix4d>> frame_info;
  
  std::string line;
  while (std::getline(info_file, line)) {
    std::istringstream iss(line);
    int id;
    double timestamp, x, y, z, qx, qy, qz, qw;
    
    if (iss >> id >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw) {
      // 构造四元数
      Eigen::Quaterniond q(qw, qx, qy, qz);
      q.normalize();
      
      // 构造变换矩阵
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      pose.block<3, 3>(0, 0) = q.toRotationMatrix();
      pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
      
      frame_info[id] = std::make_pair(timestamp, pose);
    }
  }
  info_file.close();

  // 检查目标图像ID是否存在
  if (frame_info.find(target_id) == frame_info.end()) {
    std::cerr << "目标图像ID " << target_id << " 不存在" << std::endl;
    return;
  }

  // 加载目标图像
  std::string target_filename = std::to_string(target_id) + "_" + 
                               std::to_string(static_cast<long long>(frame_info[target_id].first * 1000000)) + ".png";
  std::string target_filepath = image_path + target_filename;
  
  cv::Mat target_img = cv::imread(target_filepath);
  if (target_img.empty()) {
    std::cerr << "无法加载目标图像: " << target_filepath << std::endl;
    return;
  }
  
  target_image = RawImageData(target_img, target_id, frame_info[target_id].first, frame_info[target_id].second);

  // 清空查询图像向量
  query_images.clear();
  
  // 加载查询图像
  for (int i = 0; i < length; ++i) {
    int current_query_id = query_id + i;
    
    if (frame_info.find(current_query_id) == frame_info.end()) {
      std::cerr << "查询图像ID " << current_query_id << " 不存在，跳过" << std::endl;
      continue;
    }
    
    std::string query_filename = std::to_string(current_query_id) + "_" + 
                                std::to_string(static_cast<long long>(frame_info[current_query_id].first * 1000000)) + ".png";
    std::string query_filepath = image_path + query_filename;
    
    cv::Mat query_img = cv::imread(query_filepath);
    if (query_img.empty()) {
      std::cerr << "无法加载查询图像: " << query_filepath << std::endl;
      continue;
    }
    
    query_images.emplace_back(query_img, current_query_id, frame_info[current_query_id].first, frame_info[current_query_id].second);
  }
  
  std::cout << "成功加载目标图像 ID: " << target_id << std::endl;
  std::cout << "成功加载 " << query_images.size() << " 张查询图像" << std::endl;
}

bool LoadCalibrationConfiguration(const std::string& config_path, CalibrationData& calibration_data) {
 
    YAML::Node calibration_config = YAML::LoadFile(config_path);
    std::string calibration_version = "0";
    double calibration_car_id = -1;
    if (calibration_config["version"]) {
      calibration_version = calibration_config["version"].as<std::string>();
      std::cout << "calibration version is " << calibration_version;
    } else {
      std::cout << "calibration version is not found, use defalut value : " << calibration_version;
    }
    if (calibration_config["car_id"]) {
      calibration_car_id = calibration_config["car_id"].as<double>();
      std::cout << "calibration car id is " << calibration_car_id;
    } else {
      std::cout << "calibration car id is not found, use default value : " << calibration_car_id;
    }
  
    CameraParams camera_params;
    CameraParams right_camera_params;
    IMUParams imu_params;
    WheelParams wheel_params;
    if(calibration_config["intrinsics"]) {
      YAML::Node intrinsics = calibration_config["intrinsics"];
      if (intrinsics["camera"]) {
        YAML::Node camera_intrinsics = intrinsics["camera"];
        if(camera_intrinsics["camera_name"]){
          std::cout << "calibration camera name found ";
          camera_params.camera_name = camera_intrinsics["camera_name"].as<std::string>();
        } else if(camera_intrinsics["camera_sn"]){
          std::cout << "calibration camera sn found ";
          camera_params.camera_name = camera_intrinsics["camera_sn"].as<std::string>();
        }
        else {
          std::cout << "calibration camera sn/name  not found ";
        }
        camera_params.model_type = camera_intrinsics["model_type"].as<std::string>();
        camera_params.width = camera_intrinsics["width"].as<int>();
        camera_params.height = camera_intrinsics["height"].as<int>();
        
        camera_params.projection_parameters = camera_intrinsics["projection_parameters"].as<Eigen::Vector4d>();
        camera_params.distortion_parameters = camera_intrinsics["distortion_parameters"].as<Eigen::Vector4d>();
      } else {
        std::cout << "camera intrinsics not found!!!!!!!";
      }
  
      if (intrinsics["right_camera"]) {
        YAML::Node right_camera_intrinsics = intrinsics["right_camera"];
        right_camera_params.camera_name = right_camera_intrinsics["camera_name"].as<std::string>();
        right_camera_params.model_type = right_camera_intrinsics["model_type"].as<std::string>();
        right_camera_params.width = right_camera_intrinsics["width"].as<int>();
        right_camera_params.height = right_camera_intrinsics["height"].as<int>();
        right_camera_params.projection_parameters = right_camera_intrinsics["projection_parameters"].as<Eigen::Vector4d>();
        right_camera_params.distortion_parameters = right_camera_intrinsics["distortion_parameters"].as<Eigen::Vector4d>();
      } else {
        std::cout << "right camera intrinsics not found, use left or mono camera intrinsics";
        right_camera_params.camera_name = "right_" + camera_params.camera_name;
        right_camera_params.model_type = camera_params.model_type;
        right_camera_params.width = camera_params.width;
        right_camera_params.height = camera_params.height;
        right_camera_params.projection_parameters = camera_params.projection_parameters;
        right_camera_params.distortion_parameters = camera_params.distortion_parameters;
      }
  
      if (intrinsics["imu"]) {
        YAML::Node imu_intrinsics = intrinsics["imu"];
  
        imu_params.acc_n = imu_intrinsics["acc_n"].as<double>();
        imu_params.acc_w = imu_intrinsics["acc_w"].as<double>();
        imu_params.gyr_n = imu_intrinsics["gyr_n"].as<double>();
        imu_params.gyr_w = imu_intrinsics["gyr_w"].as<double>();
      } else {
        std::cout << "imu intrinsics not found!!!!!!!";
      }
      if (intrinsics["wheel"]) {
        YAML::Node wheel_intrinsics = intrinsics["wheel"];
  
        wheel_params.wheel_velocity_noise_sigma = wheel_intrinsics["wheel_velocity_noise_sigma"].as<double>();
        wheel_params.wheel_gyro_noise_sigma = wheel_intrinsics["wheel_gyro_noise_sigma"].as<double>();      
      } else {
        std::cout << "wheel intrinsics not found!!!!!!!";
      }
    } else {
      std::cout << "all intrinsics not found in " << config_path;
      return false;
    }
  
    Eigen::Matrix4d body_T_cam0_matrix;
    double body_T_cam0_td;
  
    Eigen::Matrix4d body_T_cam1_matrix;
    double body_T_cam1_td;
  
    // 双目基线距离默认为6cm
    double base_line = 0.06;
    
    Eigen::Matrix4d body_T_wheel_matrix;
    double body_T_wheel_td;
  
    Eigen::Matrix4d wheel_T_cam0_matrix;
    double wheel_T_cam0_td;
  
    Eigen::Matrix4d wheel_T_rtk_matrix;
    double wheel_T_rtk_td;
    
    if (calibration_config["extrinsics"]) {
      YAML::Node extrinsics = calibration_config["extrinsics"];
  
      body_T_cam0_matrix = extrinsics["body_T_cam0"].as<Eigen::Matrix4d>();
      body_T_cam0_td = extrinsics["body_T_cam0_td"].as<double>();
  
      if (extrinsics["base_line"])
      {
        base_line = extrinsics["base_line"].as<double>();
      } else
      {
        std::cout << "No base line data, maybe mono camera.";
      }
  
      if (extrinsics["body_T_cam1"])
      {
        body_T_cam1_matrix = extrinsics["body_T_cam1"].as<Eigen::Matrix4d>();
        body_T_cam1_td = extrinsics["body_T_cam1_td"].as<double>();
      }
      else
      {
        std::cout << "body_T_cam1 not exists, use base_line(6cm) to calculate right camera extrinsics.";
        body_T_cam1_matrix = body_T_cam0_matrix;
        body_T_cam1_td = body_T_cam0_td;
      }
  
      body_T_wheel_matrix = extrinsics["body_T_wheel"].as<Eigen::Matrix4d>();
      body_T_wheel_td = extrinsics["body_T_wheel_td"].as<double>();
  
  
      if (extrinsics["wheel_T_cam0"]) {
        wheel_T_cam0_matrix = extrinsics["wheel_T_cam0"].as<Eigen::Matrix4d>();
      } else {
        wheel_T_cam0_matrix = Eigen::Matrix4d::Identity();
        std::cout << "wheel_T_cam0_matrix not found";
      }
  
      if (extrinsics["wheel_T_cam0_td"]) {
        wheel_T_cam0_td = extrinsics["wheel_T_cam0_td"].as<double>();
      } else {
        wheel_T_cam0_td = 0;
      }
  
      if (extrinsics["wheel_T_rtk"]) {
        wheel_T_rtk_matrix = extrinsics["wheel_T_rtk"].as<Eigen::Matrix4d>();
      } else {
        wheel_T_rtk_matrix = Eigen::Matrix4d::Identity();
        std::cout << "wheel_T_rtk_matrix not found";
      }
  
      if (extrinsics["wheel_T_rtk_td"]) {
        wheel_T_rtk_td = extrinsics["wheel_T_rtk_td"].as<double>();
      } else {
        wheel_T_rtk_td = 0;
      }
  
      Eigen::Matrix4d wheel_T_cam1_matrix = wheel_T_cam0_matrix;
      wheel_T_cam1_matrix(1, 3) = wheel_T_cam1_matrix(1, 3) - base_line;
      body_T_cam1_matrix = body_T_wheel_matrix * wheel_T_cam1_matrix;
  
    } else {
      std::cout << "all extrinsics not found in " << config_path;
      return false;
    }
  
    ExtrinsicTransform extrinsic_body_T_cam0(body_T_cam0_matrix, body_T_cam0_td);
    ExtrinsicTransform extrinsic_body_T_cam1(body_T_cam1_matrix, body_T_cam1_td);
  
    // SLAM_LOG_HIGHLIGHT() << "!!!!!!!!!!!!!!!!!!!!!!" <<R2ypr(body_T_wheel_matrix.block<3,3>(0,0)).x();
    // Eigen::Matrix3d R = body_T_wheel_matrix.block<3,3>(0,0);
    // double yaw = R2ypr(R).x();
    // body_T_wheel_matrix.block<3,3>(0,0) = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R;
  
    // SLAM_LOG_HIGHLIGHT() << "!!!!!!!!!!!!!!!!!!!!!!" <<R2ypr(body_T_wheel_matrix.block<3,3>(0,0)).x();
  
    ExtrinsicTransform extrinsic_body_T_wheel(body_T_wheel_matrix, body_T_wheel_td);
    ExtrinsicTransform extrinsic_wheel_T_cam0(wheel_T_cam0_matrix, wheel_T_cam0_td);
    ExtrinsicTransform extrinsic_wheel_T_rtk(wheel_T_rtk_matrix, wheel_T_rtk_td);
  
    calibration_data.car_id = calibration_car_id;
    calibration_data.version = calibration_version;
    calibration_data.intrinsic_camera = camera_params;
    calibration_data.right_intrinsic_camera = right_camera_params;
    calibration_data.intrinsic_wheel = wheel_params;
    calibration_data.intrinsic_imu = imu_params;
    calibration_data.base_line = base_line;
    calibration_data.extrinsic_body_T_cam0 = extrinsic_body_T_cam0;
    calibration_data.extrinsic_body_T_cam1 = extrinsic_body_T_cam1;
    calibration_data.extrinsic_body_T_wheel = extrinsic_body_T_wheel;
    calibration_data.extrinsic_wheel_T_cam0 = extrinsic_wheel_T_cam0;
    calibration_data.extrinsic_wheel_T_rtk = extrinsic_wheel_T_rtk;
  
    return true;
}

bool LoadMainConfiguration(const std::string& config_path, MainConfig& config) {
    try {
        YAML::Node config_file = YAML::LoadFile(config_path);
        
        // Load data parameters
        if (config_file["data"]) {
            const auto& data_node = config_file["data"];
            config.target_image_id = data_node["target_image_id"].as<int>(config.target_image_id);
            config.query_image_start_id = data_node["query_image_start_id"].as<int>(config.query_image_start_id);
            config.query_image_count = data_node["query_image_count"].as<int>(config.query_image_count);
            config.query_image_length = data_node["query_image_length"].as<int>(config.query_image_length);
        }
        
        // Load debug options
        if (config_file["debug"]) {
            const auto& debug_node = config_file["debug"];
            config.enable_debug = debug_node["enable_debug"].as<bool>(config.enable_debug);
            config.save_intermediate = debug_node["save_intermediate"].as<bool>(config.save_intermediate);
            config.output_dir = debug_node["output_dir"].as<std::string>(config.output_dir);
            config.dr_debug = debug_node["dr_debug"].as<bool>(config.dr_debug);
        }
        
        // Load file paths
        if (config_file["paths"]) {
            const auto& paths_node = config_file["paths"];
            config.calibration_config_path = paths_node["calibration_config"].as<std::string>(config.calibration_config_path);
            config.output_prefix = paths_node["output_prefix"].as<std::string>(config.output_prefix);
            
            // Load data file paths
            config.gnss_file = paths_node["gnss_file"].as<std::string>(config.gnss_file);
            config.imu_file = paths_node["imu_file"].as<std::string>(config.imu_file);
            config.odom_file = paths_node["odom_file"].as<std::string>(config.odom_file);
        }
        
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading configuration file: " << e.what() << std::endl;
        return false;
    }
}

