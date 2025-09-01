#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include "data_prepare.hpp"


int main(int argc, char** argv) {
    std::cout << GetConfigPath() << std::endl;
    CalibrationData calibration_data;
    LoadCalibrationConfiguration(GetConfigPath(), calibration_data);
    std::cout << calibration_data.car_id << std::endl;
    std::cout << calibration_data.version << std::endl;
    std::cout << calibration_data.intrinsic_camera.model_type << std::endl;
    std::cout << calibration_data.intrinsic_camera.camera_name << std::endl;
    std::cout << calibration_data.intrinsic_camera.scaling_ratio << std::endl;
    std::cout << calibration_data.intrinsic_camera.width << std::endl;
    std::cout << calibration_data.intrinsic_camera.height << std::endl;
    
    
    RawImageData target_image;
    std::vector<RawImageData> query_images;
    LoadRawImageData(10,10,10,target_image,query_images);
    return 0;
}