#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include "data_prepare.hpp"
#include "slam/feature_extractors.hpp"
#include "slam/feature_trackers.hpp"
#include "slam/depth_estimators.hpp"
#include "slam/bundle_adjusters.hpp"

using namespace slam;

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

void drawFeatures(cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
                  const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
    for (const auto& kp : keypoints) {
        cv::circle(image, kp.pt, 3, color, -1);
        cv::circle(image, kp.pt, 5, color, 1);
    }
}

void drawTracks(cv::Mat& image, const std::vector<cv::Point2f>& prev_points,
                const std::vector<cv::Point2f>& curr_points,
                const std::vector<uchar>& status) {
    for (size_t i = 0; i < prev_points.size(); ++i) {
        if (status[i]) {
            cv::line(image, prev_points[i], curr_points[i], cv::Scalar(0, 255, 0), 1);
            cv::circle(image, curr_points[i], 3, cv::Scalar(0, 0, 255), -1);
        }
    }
}

void drawReprojection(cv::Mat& image, const std::vector<cv::Point2f>& observed_points,
                     const std::vector<cv::Point2f>& projected_points,
                     double threshold = 2.0) {
    for (size_t i = 0; i < observed_points.size(); ++i) {
        double error = cv::norm(observed_points[i] - projected_points[i]);
        cv::Scalar color = (error < threshold) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::line(image, observed_points[i], projected_points[i], color, 1);
        cv::circle(image, observed_points[i], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(image, projected_points[i], 2, color, -1);
    }
}

void printResults(const std::string& module_name, 
                 const std::vector<cv::KeyPoint>& keypoints,
                 double time_ms,
                 const std::string& additional_info = "") {
    std::cout << "\n=== " << module_name << " Results ===" << std::endl;
    std::cout << "Number of keypoints: " << keypoints.size() << std::endl;
    std::cout << "Processing time: " << time_ms << " ms" << std::endl;
    if (!additional_info.empty()) {
        std::cout << additional_info << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

void printResults(const std::string& module_name,
                 const FeatureTracker::TrackingResult& track_result,
                 double time_ms,
                 const std::string& additional_info = "") {
    std::cout << "\n=== " << module_name << " Results ===" << std::endl;
    std::cout << "Tracked points: " << track_result.tracked_count << "/" << track_result.prev_points.size() << std::endl;
    std::cout << "Processing time: " << time_ms << " ms" << std::endl;
    std::cout << "Average error: " << track_result.avg_error << " pixels" << std::endl;
    if (!additional_info.empty()) {
        std::cout << additional_info << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

void printResults(const std::string& module_name,
                 const DepthEstimator::DepthResult& depth_result,
                 double time_ms,
                 const std::string& additional_info = "") {
    std::cout << "\n=== " << module_name << " Results ===" << std::endl;
    std::cout << "Valid 3D points: " << depth_result.valid_points_count << "/" << depth_result.points_3d.size() << std::endl;
    std::cout << "Processing time: " << time_ms << " ms" << std::endl;
    std::cout << "Average depth: " << depth_result.avg_depth << " m" << std::endl;
    if (!additional_info.empty()) {
        std::cout << additional_info << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

void printResults(const std::string& module_name,
                 const BundleAdjuster::BAResult& ba_result,
                 double time_ms,
                 const std::string& additional_info = "") {
    std::cout << "\n=== " << module_name << " Results ===" << std::endl;
    std::cout << "Initial error: " << ba_result.initial_error << std::endl;
    std::cout << "Final error: " << ba_result.final_error << std::endl;
    std::cout << "Iterations: " << ba_result.iterations << std::endl;
    std::cout << "Processing time: " << time_ms << " ms" << std::endl;
    if (!additional_info.empty()) {
        std::cout << additional_info << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

int main() {
    std::cout << "SLAM Visual Pipeline Demo with Real Images" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Load configuration
    std::string config_path = GetMainConfigPath();
    std::cout << "Loading configuration from: " << config_path << std::endl;
    
    MainConfig config;
    if (!LoadMainConfiguration(config_path, config)) {
        std::cerr << "Failed to load configuration file" << std::endl;
        return 1;
    }
    
    // Load calibration data
    std::cout << "Loading calibration data from: " << config.calibration_config_path << std::endl;
    CalibrationData calibration_data;
    if (!LoadCalibrationConfiguration(config.calibration_config_path, calibration_data)) {
        std::cerr << "Failed to load calibration data" << std::endl;
        return 1;
    }
    
    std::cout << "Calibration data loaded successfully" << std::endl;
    std::cout << "Camera: " << calibration_data.intrinsic_camera.camera_name << std::endl;
    std::cout << "Resolution: " << calibration_data.intrinsic_camera.width << "x" << calibration_data.intrinsic_camera.height << std::endl;
    
    // Load real images
    std::vector<cv::Mat> images;
    
    // Load first image
    cv::Mat img1 = cv::imread("/home/watermango/data/raw_data_for_loop_closure/image/10_1743404455100000.png", cv::IMREAD_COLOR);
    if (img1.empty()) {
        std::cerr << "Failed to load first image" << std::endl;
        return 1;
    }
    
    // Load second image
    cv::Mat img2 = cv::imread("/home/watermango/data/raw_data_for_loop_closure/image/11_1743404455300000.png", cv::IMREAD_COLOR);
    if (img2.empty()) {
        std::cerr << "Failed to load second image" << std::endl;
        return 1;
    }
    
    images.push_back(img1.clone());
    images.push_back(img2.clone());
    
    std::cout << "Loaded " << images.size() << " images" << std::endl;
    
    // Extract camera intrinsics
    std::cout << "Projection parameters size: " << calibration_data.intrinsic_camera.projection_parameters.size() << std::endl;
    std::cout << "Distortion parameters size: " << calibration_data.intrinsic_camera.distortion_parameters.size() << std::endl;
    
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        calibration_data.intrinsic_camera.projection_parameters[0], 0, calibration_data.intrinsic_camera.projection_parameters[2],
        0, calibration_data.intrinsic_camera.projection_parameters[1], calibration_data.intrinsic_camera.projection_parameters[3],
        0, 0, 1);
    
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    if (calibration_data.intrinsic_camera.distortion_parameters.size() >= 5) {
        dist_coeffs.at<double>(0) = calibration_data.intrinsic_camera.distortion_parameters[0];
        dist_coeffs.at<double>(1) = calibration_data.intrinsic_camera.distortion_parameters[1];
        dist_coeffs.at<double>(2) = calibration_data.intrinsic_camera.distortion_parameters[2];
        dist_coeffs.at<double>(3) = calibration_data.intrinsic_camera.distortion_parameters[3];
        dist_coeffs.at<double>(4) = calibration_data.intrinsic_camera.distortion_parameters[4];
    }
    
    // Set up debug options
    ModuleBase::DebugOptions debug_opts;
    debug_opts.enable_visualization = true;
    debug_opts.save_intermediate = true;
    debug_opts.output_dir = "./visual_demo_output";
    debug_opts.wait_key_delay = 2000;
    
    // Create output directory
    std::filesystem::create_directories(debug_opts.output_dir);
    
    // Display original images
    cv::Mat display1 = images[0].clone();
    cv::Mat display2 = images[1].clone();
    
    cv::putText(display1, "Frame 1", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                cv::Scalar(255, 255, 255), 2);
    cv::putText(display2, "Frame 2", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                cv::Scalar(255, 255, 255), 2);
    
    cv::imshow("Original Images", display1);
    cv::waitKey(1000);
    cv::imshow("Original Images", display2);
    cv::waitKey(1000);
    
    // 1. Feature Extraction
    std::cout << "\n1. Feature Extraction..." << std::endl;
    
    FeatureExtractor::Options feat_opts;
    feat_opts.max_features = 1000;
    
    debug_opts.module_name = "feature_extraction";
    auto extractor = createFeatureExtractor("ORB", feat_opts);
    extractor->setDebugOptions(debug_opts);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto features = extractor->extract(images[0]);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Draw features
    cv::Mat feat_display = display1.clone();
    drawFeatures(feat_display, features.keypoints);
    cv::putText(feat_display, "Features: " + std::to_string(features.keypoints.size()), 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Feature Extraction", feat_display);
    cv::waitKey(debug_opts.wait_key_delay);
    
    printResults("Feature Extraction (ORB)", features.keypoints, duration.count());
    
    // 2. Feature Tracking
    std::cout << "\n2. Feature Tracking..." << std::endl;
    
    FeatureTracker::Options track_opts;
    track_opts.max_level = 3;
    track_opts.win_size = cv::Size(21, 21);
    
    debug_opts.module_name = "feature_tracking";
    auto tracker = createFeatureTracker("LK", track_opts);
    tracker->setDebugOptions(debug_opts);
    
    // Convert features to points for tracking
    std::vector<cv::Point2f> track_points;
    for (const auto& kp : features.keypoints) {
        track_points.push_back(kp.pt);
    }
    
    start = std::chrono::high_resolution_clock::now();
    auto track_result = tracker->track(images[0], images[1], track_points);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Draw tracks
    cv::Mat track_display = display2.clone();
    drawTracks(track_display, track_result.prev_points, track_result.current_points, track_result.status);
    cv::putText(track_display, "Tracked: " + std::to_string(track_result.tracked_count) + "/" + std::to_string(track_points.size()), 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Feature Tracking", track_display);
    cv::waitKey(debug_opts.wait_key_delay);
    
    printResults("Feature Tracking (LK)", track_result, duration.count());
    
    // 3. Depth Estimation
    std::cout << "\n3. Depth Estimation..." << std::endl;
    
    DepthEstimator::Options depth_opts;
    depth_opts.min_depth = 0.1;
    depth_opts.max_depth = 50.0;
    depth_opts.min_parallax_deg = 1.0;
    depth_opts.max_parallax_deg = 45.0;
    
    debug_opts.module_name = "depth_estimation";
    auto estimator = createDepthEstimator("Optimal", depth_opts);
    estimator->setDebugOptions(debug_opts);
    
    // Use matched points from tracking
    std::vector<cv::Point2f> valid_points1, valid_points2;
    for (size_t i = 0; i < track_result.prev_points.size(); ++i) {
        if (track_result.status[i]) {
            valid_points1.push_back(track_result.prev_points[i]);
            valid_points2.push_back(track_result.current_points[i]);
        }
    }
    
    // Get wheel to camera transformation from calibration data
    Eigen::Matrix4d T_wheel_cam = calibration_data.extrinsic_wheel_T_cam0.transform;
    std::cout << "Wheel to camera transformation loaded" << std::endl;
    
    // Load wheel poses for the two images (simplified - in real scenario these would come from sensor data)
    Eigen::Matrix4d wheel_pose1 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d wheel_pose2 = Eigen::Matrix4d::Identity();
    wheel_pose2(0, 3) = 1.0;  // Move 1m forward in wheel coordinates
    
    // Convert wheel poses to camera poses using the same method as SFMReconstructor
    auto wheelPoseToCamPose = [&T_wheel_cam](const Eigen::Matrix4d& T_w_wheel) -> Eigen::Matrix4d {
        return T_w_wheel * T_wheel_cam;
    };
    
    Eigen::Matrix4d pose1 = wheelPoseToCamPose(wheel_pose1);
    Eigen::Matrix4d pose2 = wheelPoseToCamPose(wheel_pose2);
    
    std::cout << "Camera pose 1 translation: " << pose1.block<3, 1>(0, 3).transpose() << std::endl;
    std::cout << "Camera pose 2 translation: " << pose2.block<3, 1>(0, 3).transpose() << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    auto depth_result = estimator->triangulate(
        valid_points1, valid_points2,
        pose1, pose2,
        camera_matrix, dist_coeffs,
        images[0], images[1]
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Draw triangulated points with depth-based coloring
    cv::Mat depth_display = display2.clone();
    for (size_t i = 0; i < valid_points2.size() && i < depth_result.points_3d.size(); ++i) {
        const auto& pt_3d = depth_result.points_3d[i];
        if (pt_3d.x != 0 || pt_3d.y != 0 || pt_3d.z != 0) {
            double depth = pt_3d.z;
            // Color based on depth (red=far, green=near)
            double normalized_depth = std::min(depth / 10.0, 1.0);
            cv::Scalar color(255 * normalized_depth, 255 * (1 - normalized_depth), 0);
            cv::circle(depth_display, valid_points2[i], 5, color, -1);
        }
    }
    
    cv::putText(depth_display, "3D Points: " + std::to_string(depth_result.valid_points_count), 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Depth Estimation", depth_display);
    cv::waitKey(debug_opts.wait_key_delay);
    
    printResults("Depth Estimation (Optimal)", depth_result, duration.count());
    
    // 4. Bundle Adjustment
    std::cout << "\n4. Bundle Adjustment..." << std::endl;
    
    BundleAdjuster::Options ba_opts;
    ba_opts.max_iterations = 20;
    ba_opts.use_robust_loss = true;
    ba_opts.huber_parameter = 1.0;
    ba_opts.verbose = true;
    
    debug_opts.module_name = "bundle_adjustment";
    auto ba = createBundleAdjuster("Ceres", ba_opts);
    ba->setDebugOptions(debug_opts);
    
    // Create BA problem
    BundleAdjuster::BAProblem problem;
    
    // Add camera poses (with some noise)
    for (int i = 0; i < 2; ++i) {
        Eigen::Matrix4d pose = (i == 0) ? pose1 : pose2;
        // Add small noise
        pose(0, 3) += 0.05 * (rand() % 200 - 100) / 100.0;
        pose(1, 3) += 0.05 * (rand() % 200 - 100) / 100.0;
        problem.poses.push_back(pose);
        problem.pose_fixed.push_back(i == 0);  // Fix first pose
    }
    
    // Add 3D points
    for (const auto& pt : depth_result.points_3d) {
        if (pt.x != 0 || pt.y != 0 || pt.z != 0) {
            Eigen::Vector3d pt_3d(pt.x, pt.y, pt.z);
            // Add small noise
            pt_3d(0) += 0.02 * (rand() % 200 - 100) / 100.0;
            pt_3d(1) += 0.02 * (rand() % 200 - 100) / 100.0;
            pt_3d(2) += 0.02 * (rand() % 200 - 100) / 100.0;
            problem.points.push_back(pt_3d);
        }
    }
    
    // Add observations
    int point_idx = 0;
    for (size_t i = 0; i < valid_points1.size(); ++i) {
        if (i < depth_result.points_3d.size() && 
            (depth_result.points_3d[i].x != 0 || 
             depth_result.points_3d[i].y != 0 || 
             depth_result.points_3d[i].z != 0)) {
            
            problem.observations.push_back({0, point_idx});
            problem.measurements.push_back(valid_points1[i]);
            
            problem.observations.push_back({1, point_idx});
            problem.measurements.push_back(valid_points2[i]);
            
            point_idx++;
        }
    }
    
    problem.camera_matrix = camera_matrix;
    problem.dist_coeffs = dist_coeffs;
    
    start = std::chrono::high_resolution_clock::now();
    auto ba_result = ba->optimize(problem);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Visualize reprojection errors
    cv::Mat ba_display = display2.clone();
    std::vector<cv::Point2f> projected_points;
    
    // Project optimized points to image for second frame
    std::vector<cv::Point2f> observed_points;
    for (size_t i = 0; i < problem.observations.size(); ++i) {
        const auto& obs = problem.observations[i];
        if (obs.first == 1) {  // Only show for second frame
            const Eigen::Vector3d& pt_3d = problem.points[obs.second];
            const Eigen::Matrix4d& pose = problem.poses[obs.first];
            
            // Transform to camera coordinates
            Eigen::Vector4d pt_cam = pose.inverse() * Eigen::Vector4d(pt_3d(0), pt_3d(1), pt_3d(2), 1.0);
            
            // Project to image
            cv::Mat_<double> pt_cam_mat(3, 1);
            pt_cam_mat(0) = pt_cam(0) / pt_cam(2);
            pt_cam_mat(1) = pt_cam(1) / pt_cam(2);
            pt_cam_mat(2) = 1.0;
            
            cv::Mat_<double> projected = camera_matrix * pt_cam_mat;
            projected_points.push_back(cv::Point2f(projected(0), projected(1)));
            observed_points.push_back(problem.measurements[i]);
        }
    }
    
    drawReprojection(ba_display, observed_points, projected_points);
    cv::putText(ba_display, "BA Error: " + std::to_string(ba_result.final_error), 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Bundle Adjustment", ba_display);
    cv::waitKey(debug_opts.wait_key_delay);
    
    printResults("Bundle Adjustment (Ceres)", ba_result, duration.count());
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Pipeline Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Feature Extraction: " << features.keypoints.size() << " features in " << features.extraction_time_ms << " ms" << std::endl;
    std::cout << "Feature Tracking: " << track_result.tracked_count << "/" 
              << track_points.size() << " points tracked in " << track_result.tracking_time_ms << " ms" << std::endl;
    std::cout << "Depth Estimation: " << depth_result.valid_points_count << " 3D points in " << depth_result.triangulation_time_ms << " ms" << std::endl;
    std::cout << "Bundle Adjustment: Error reduced from " << ba_result.initial_error 
              << " to " << ba_result.final_error << " in " << ba_result.optimization_time_ms << " ms" << std::endl;
    std::cout << "Total Processing Time: " 
              << features.extraction_time_ms + track_result.tracking_time_ms + 
                 depth_result.triangulation_time_ms + ba_result.optimization_time_ms
              << " ms" << std::endl;
    std::cout << "\nDebug outputs saved to: " << debug_opts.output_dir << std::endl;
    
    // Wait for user input
    std::cout << "\nPress any key to close visualization windows..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}