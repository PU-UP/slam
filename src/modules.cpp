#include "slam/modules.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>

namespace slam {

void ModuleBase::visualizeImage(const std::string& window_name, const cv::Mat& image) const {
    if (!debug_options_.enable_visualization) return;
    
    if (image.empty()) {
        std::cerr << "Warning: Cannot visualize empty image for window: " << window_name << std::endl;
        return;
    }
    
    cv::Mat display = image.clone();
    if (debug_options_.scale_factor != 1.0) {
        cv::resize(display, display, cv::Size(), debug_options_.scale_factor, 
                  debug_options_.scale_factor);
    }
    
    // 创建窗口并显示图片
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, display);
    
    // 等待按键或超时
    cv::waitKey(debug_options_.wait_key_delay);
    
    // 如果wait_key_delay为0，等待任意按键
    if (debug_options_.wait_key_delay == 0) {
        std::cout << "Press any key to close the window: " << window_name << std::endl;
        cv::waitKey(0);
    }
    
    // 关闭窗口
    cv::destroyWindow(window_name);
}

void ModuleBase::saveImage(const std::string& filename, const cv::Mat& image) const {
    if (!debug_options_.save_intermediate) return;
    
    std::string full_path = getOutputPath(filename);
    cv::imwrite(full_path, image);
}

void ModuleBase::saveData(const std::string& filename, const std::string& data) const {
    if (!debug_options_.save_intermediate) return;
    
    std::string full_path = getOutputPath(filename);
    std::ofstream out(full_path);
    out << data;
    out.close();
}

std::string ModuleBase::getOutputPath(const std::string& filename) const {
    std::filesystem::create_directories(debug_options_.output_dir);
    return debug_options_.output_dir + "/" + debug_options_.module_name + "_" + filename;
}

// FeatureExtractor visualization
void FeatureExtractor::visualize(const Features& features) const {
    if (!debug_options_.enable_visualization) return;
    
    cv::Mat display;
    cv::drawKeypoints(features.image, features.keypoints, display,
                     cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    
    if (debug_options_.show_text) {
        std::string text = "Features: " + std::to_string(features.keypoints.size());
        if (features.extraction_time_ms > 0) {
            text += " | Time: " + std::to_string(features.extraction_time_ms) + "ms";
        }
        cv::putText(display, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
    
    visualizeImage("Feature Extraction", display);
    saveImage("features.png", display);
    
    // Save feature data
    if (debug_options_.save_intermediate) {
        std::string data = "feature_id,x,y,angle,octave,response\n";
        for (size_t i = 0; i < features.keypoints.size(); ++i) {
            const auto& kp = features.keypoints[i];
            data += std::to_string(i) + "," + 
                   std::to_string(kp.pt.x) + "," + 
                   std::to_string(kp.pt.y) + "," +
                   std::to_string(kp.angle) + "," +
                   std::to_string(kp.octave) + "," +
                   std::to_string(kp.response) + "\n";
        }
        saveData("features.csv", data);
    }
}

// FeatureTracker visualization
void FeatureTracker::visualize(const TrackingResult& result) const {
    if (!debug_options_.enable_visualization) return;
    
    cv::Mat display;
    cv::hconcat(result.prev_image, result.current_image, display);
    
    // Draw tracks
    for (size_t i = 0; i < result.prev_points.size(); ++i) {
        if (result.status[i]) {
            cv::Point2f prev_pt = result.prev_points[i];
            cv::Point2f curr_pt = result.current_points[i] + 
                                cv::Point2f(result.prev_image.cols, 0);
            
            // Color based on error
            double error = result.errors[i];
            cv::Scalar color;
            if (error < 1.0) color = cv::Scalar(0, 255, 0);  // Green
            else if (error < 5.0) color = cv::Scalar(0, 255, 255);  // Yellow
            else color = cv::Scalar(0, 0, 255);  // Red
            
            cv::line(display, prev_pt, curr_pt, color, 1);
            cv::circle(display, prev_pt, 2, color, -1);
            cv::circle(display, curr_pt, 2, color, -1);
        }
    }
    
    if (debug_options_.show_text) {
        std::string text = "Tracked: " + std::to_string(result.tracked_count) + "/" +
                          std::to_string(result.prev_points.size());
        if (result.tracking_time_ms > 0) {
            text += " | Time: " + std::to_string(result.tracking_time_ms) + "ms";
        }
        text += " | Avg Error: " + std::to_string(result.avg_error);
        cv::putText(display, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
    
    visualizeImage("Feature Tracking", display);
    saveImage("tracking.png", display);
    
    // Save tracking data
    if (debug_options_.save_intermediate) {
        std::string data = "track_id,prev_x,prev_y,curr_x,curr_y,status,error\n";
        for (size_t i = 0; i < result.prev_points.size(); ++i) {
            data += std::to_string(result.track_ids[i]) + "," +
                   std::to_string(result.prev_points[i].x) + "," +
                   std::to_string(result.prev_points[i].y) + "," +
                   std::to_string(result.current_points[i].x) + "," +
                   std::to_string(result.current_points[i].y) + "," +
                   std::to_string(static_cast<int>(result.status[i])) + "," +
                   std::to_string(result.errors[i]) + "\n";
        }
        saveData("tracking.csv", data);
    }
}

// PoseEstimator visualization
void PoseEstimator::visualize(const PoseResult& result) const {
    if (!debug_options_.enable_visualization || result.image.empty()) return;
    
    cv::Mat display = result.image.clone();
    
    // Draw inlier matches
    for (size_t i = 0; i < result.inlier_points_2d.size(); ++i) {
        cv::circle(display, result.inlier_points_2d[i], 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // Draw coordinate axes at camera center
    cv::Point2f center(display.cols / 2, display.rows / 2);
    float axis_length = 50.0f;
    
    // X-axis (red)
    cv::Point2f x_end = center + cv::Point2f(axis_length, 0);
    cv::arrowedLine(display, center, x_end, cv::Scalar(0, 0, 255), 2);
    
    // Y-axis (green)
    cv::Point2f y_end = center - cv::Point2f(0, axis_length);
    cv::arrowedLine(display, center, y_end, cv::Scalar(0, 255, 0), 2);
    
    if (debug_options_.show_text) {
        std::string text = "Inliers: " + std::to_string(result.inlier_count) + "/" +
                          std::to_string(result.inlier_points_2d.size());
        text += " | Reproj Error: " + std::to_string(result.reprojection_error);
        text += " | Confidence: " + std::to_string(result.confidence);
        if (result.estimation_time_ms > 0) {
            text += " | Time: " + std::to_string(result.estimation_time_ms) + "ms";
        }
        cv::putText(display, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
    
    visualizeImage("Pose Estimation", display);
    saveImage("pose_estimation.png", display);
    
    // Save pose data
    if (debug_options_.save_intermediate) {
        std::string data = "pose_matrix:\n";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                data += std::to_string(result.pose(i, j));
                if (j < 3) data += ", ";
            }
            data += "\n";
        }
        data += "\nmetrics:\n";
        data += "inlier_count," + std::to_string(result.inlier_count) + "\n";
        data += "reprojection_error," + std::to_string(result.reprojection_error) + "\n";
        data += "confidence," + std::to_string(result.confidence) + "\n";
        saveData("pose.txt", data);
    }
}

// DepthEstimator visualization
void DepthEstimator::visualize(const DepthResult& result) const {
    if (!debug_options_.enable_visualization) return;
    
    cv::Mat display;
    cv::hconcat(result.reference_image, result.current_image, display);
    
    // Draw triangulated points with depth-based coloring
    double max_depth = result.avg_depth * 2.0;
    for (size_t i = 0; i < result.points_2d_ref.size(); ++i) {
        if (result.is_valid[i]) {
            cv::Point2f ref_pt = result.points_2d_ref[i];
            cv::Point2f cur_pt = result.points_2d_cur[i] + 
                                cv::Point2f(result.reference_image.cols, 0);
            
            // Color based on depth
            double depth_ratio = std::min(result.depths[i] / max_depth, 1.0);
            cv::Scalar color(0, 255 * (1 - depth_ratio), 255 * depth_ratio);
            
            cv::line(display, ref_pt, cur_pt, color, 1);
            cv::circle(display, ref_pt, 2, color, -1);
            cv::circle(display, cur_pt, 2, color, -1);
        }
    }
    
    if (debug_options_.show_text) {
        std::string text = "Valid Points: " + std::to_string(result.valid_points_count) + "/" +
                          std::to_string(result.points_2d_ref.size());
        text += " | Avg Depth: " + std::to_string(result.avg_depth);
        text += " | Avg Parallax: " + std::to_string(result.avg_parallax);
        if (result.triangulation_time_ms > 0) {
            text += " | Time: " + std::to_string(result.triangulation_time_ms) + "ms";
        }
        cv::putText(display, text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
    
    visualizeImage("Depth Estimation", display);
    saveImage("depth_estimation.png", display);
    
    // Save depth data
    if (debug_options_.save_intermediate) {
        std::string data = "point_id,x1,y1,x2,y2,depth,parallax,is_valid\n";
        for (size_t i = 0; i < result.points_2d_ref.size(); ++i) {
            data += std::to_string(i) + "," +
                   std::to_string(result.points_2d_ref[i].x) + "," +
                   std::to_string(result.points_2d_ref[i].y) + "," +
                   std::to_string(result.points_2d_cur[i].x) + "," +
                   std::to_string(result.points_2d_cur[i].y) + "," +
                   std::to_string(result.depths[i]) + "," +
                   std::to_string(result.parallax_angles[i]) + "," +
                   std::to_string(result.is_valid[i]) + "\n";
        }
        saveData("depth.csv", data);
    }
}

// BundleAdjuster visualization
void BundleAdjuster::visualize(const BAResult& result, const BAProblem& /*problem*/) const {
    if (!debug_options_.enable_visualization) return;
    
    // Create error plot
    const int width = 800;
    const int height = 400;
    cv::Mat error_plot(height, width, CV_8UC3, cv::Scalar(20, 20, 20));
    
    // Draw axes
    cv::line(error_plot, cv::Point(50, height - 50), 
             cv::Point(width - 50, height - 50), cv::Scalar(200, 200, 200), 2);
    cv::line(error_plot, cv::Point(50, 50), 
             cv::Point(50, height - 50), cv::Scalar(200, 200, 200), 2);
    
    // Draw error evolution
    double max_error = std::max(result.initial_error, result.final_error) * 1.1;
    cv::Point2f start(50, height - 50);
    cv::Point2f end(width - 50, height - 50);
    
    // Initial error
    cv::Point2f initial_pt(
        50, 
        height - 50 - (result.initial_error / max_error) * (height - 100)
    );
    cv::circle(error_plot, initial_pt, 5, cv::Scalar(0, 0, 255), -1);
    
    // Final error
    cv::Point2f final_pt(
        width - 50, 
        height - 50 - (result.final_error / max_error) * (height - 100)
    );
    cv::circle(error_plot, final_pt, 5, cv::Scalar(0, 255, 0), -1);
    
    // Draw connecting line
    cv::line(error_plot, initial_pt, final_pt, cv::Scalar(100, 100, 100), 2);
    
    // Add labels
    cv::putText(error_plot, "Initial Error: " + std::to_string(result.initial_error),
               cv::Point(60, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    cv::putText(error_plot, "Final Error: " + std::to_string(result.final_error),
               cv::Point(60, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    cv::putText(error_plot, "Iterations: " + std::to_string(result.iterations),
               cv::Point(width - 200, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    
    if (result.optimization_time_ms > 0) {
        cv::putText(error_plot, "Time: " + std::to_string(result.optimization_time_ms) + "ms",
                   cv::Point(width - 200, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    }
    
    visualizeImage("Bundle Adjustment Error", error_plot);
    saveImage("ba_error.png", error_plot);
    
    // Save BA data
    if (debug_options_.save_intermediate) {
        std::string data = "bundle_adjustment_results:\n";
        data += "initial_error," + std::to_string(result.initial_error) + "\n";
        data += "final_error," + std::to_string(result.final_error) + "\n";
        data += "improvement_ratio," + std::to_string(
            (result.initial_error - result.final_error) / result.initial_error) + "\n";
        data += "iterations," + std::to_string(result.iterations) + "\n";
        data += "success," + std::to_string(result.success) + "\n";
        data += "optimization_time_ms," + std::to_string(result.optimization_time_ms) + "\n";
        saveData("ba_results.txt", data);
    }
}

} // namespace slam