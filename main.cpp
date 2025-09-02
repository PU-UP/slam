#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include "data_prepare.hpp"
#include "sfm_reconstructor.hpp"


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
    LoadRawImageData(10, 10, 10, target_image, query_images);


    SFMOptions opts; 
    opts.enable_ba = true; 
    opts.ba_max_iterations = 80; 
    opts.prior_trans_sigma = 0.05; 
    opts.prior_rot_sigma_rad = 2.0*M_PI/180.0;
    
    SFMReconstructor recon(calibration_data, opts);


    std::cout << "Start reconstruction" << std::endl;
    // 2) 重建 + BA
    SFMResult result = recon.Reconstruct(query_images);

    std::cout << "End reconstruction" << std::endl;

    return 0;
}