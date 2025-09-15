#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include "data_prepare.hpp"
#include "include/slam/modules.hpp"

// Configuration file path finder
inline std::string GetMainConfigPath() {
    // Try relative path
    std::string relative_path = "config.yaml";
    if (std::filesystem::exists(relative_path)) {
        return relative_path;
    }
    
    // Try parent directory
    std::string parent_path = "../config.yaml";
    if (std::filesystem::exists(parent_path)) {
        return parent_path;
    }
    
    // Use absolute path as fallback
    return "/home/watermango/github/slam/config.yaml";
}

int main(int argc, char** argv) {
    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        config_path = GetMainConfigPath();
    }
    // Load main configuration
    std::cout << "Loading configuration from: " << config_path << std::endl;
    
    MainConfig config;
    if (!LoadMainConfiguration(config_path, config)) {
        std::cerr << "Failed to load configuration file" << std::endl;
        return 1;
    }
    
    std::cout << "Configuration loaded successfully" << std::endl;
    
    // Load calibration data
    std::cout << "Loading calibration data from: " << config.calibration_config_path << std::endl;
    CalibrationData calibration_data;
    if (!LoadCalibrationConfiguration(config.calibration_config_path, calibration_data)) {
        std::cerr << "Failed to load calibration data" << std::endl;
        return 1;
    }
    
    if (config.enable_debug) {
        std::cout << "Calibration data loaded:" << std::endl;
        std::cout << "  Car ID: " << calibration_data.car_id << std::endl;
        std::cout << "  Version: " << calibration_data.version << std::endl;
        std::cout << "  Camera: " << calibration_data.intrinsic_camera.camera_name << std::endl;
        std::cout << "  Resolution: " << calibration_data.intrinsic_camera.width << "x" << calibration_data.intrinsic_camera.height << std::endl;
    }
    
    // Load image data
    RawImageData target_image;
    std::vector<RawImageData> query_images;
    LoadRawImageData(config.target_image_id, config.query_image_start_id, 
                     config.query_image_length, target_image, query_images);
    
    if (config.enable_debug) {
        std::cout << "Loaded " << query_images.size() << " query images" << std::endl;
        std::cout << "Target image ID: " << target_image.id << std::endl;
    }

    using namespace bagio;

    TxtDataLoader::Options options;
    options.gnss_file = "../data/gnss_data.txt";
    options.imu_file = "../data/bmi_imu_data.txt";
    options.odom_file = "../data/odom_data.txt";

    TxtDataLoader data_loader(options);
    data_loader.load();

    std::cout << "Loaded counts: imu=" << data_loader.imuCount()
              << " odom=" << data_loader.odomCount()
              << " gnss=" << data_loader.gnssCount() << "\n";
    std::cout << "Time range: [" << std::to_string(data_loader.startTime())
              << ", " << std::to_string(data_loader.endTime())  
              << "]  duration=" << data_loader.duration() << " s\n";

    

    return 0;
}