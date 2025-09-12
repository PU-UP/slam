#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "slam/feature_extractors.hpp"
#include "slam/feature_trackers.hpp"

using namespace slam;

void printUsage() {
    std::cout << "Usage: test_feature_tracker <video_path> [tracker_type] [feature_type]" << std::endl;
    std::cout << "Tracker types: LK, Matching, PyramidLK, SparseFlow, Deep" << std::endl;
    std::cout << "Feature types: ORB, ShiTomasi, SIFT, SURF, FAST_BRIEF" << std::endl;
    std::cout << "Example: ./test_feature_tracker ../video.mp4 LK ORB" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string video_path = argv[1];
    std::string tracker_type = (argc > 2) ? argv[2] : "LK";
    std::string feature_type = (argc > 3) ? argv[3] : "ORB";
    
    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << std::endl;
        return 1;
    }
    
    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video info:" << std::endl;
    std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    std::cout << "  Using tracker: " << tracker_type << std::endl;
    std::cout << "  Using features: " << feature_type << std::endl;
    
    // Create feature extractor and tracker
    FeatureExtractor::Options feat_options;
    feat_options.max_features = 500;  // Reduced for better visualization
    
    auto extractor = createFeatureExtractor(feature_type, feat_options);
    
    FeatureTracker::Options track_options;
    track_options.max_level = 3;
    track_options.win_size = cv::Size(21, 21);
    track_options.max_error = 30.0;
    
    auto tracker = createFeatureTracker(tracker_type, track_options);
    
    // Set feature extractor for matching-based trackers
    if (tracker_type == "Matching") {
        auto matching_tracker = dynamic_cast<FeatureMatchingTracker*>(tracker.get());
        if (matching_tracker) {
            std::shared_ptr<FeatureExtractor> shared_extractor(extractor.release());
            matching_tracker->setFeatureExtractor(shared_extractor);
        }
    }
    
    // Set debug options
    ModuleBase::DebugOptions debug_opts;
    debug_opts.enable_visualization = true;
    debug_opts.save_intermediate = true;
    debug_opts.output_dir = "./debug_tracking";
    debug_opts.module_name = tracker_type + "_tracker";
    debug_opts.wait_key_delay = 1;  // Small delay for video
    
    extractor->setDebugOptions(debug_opts);
    tracker->setDebugOptions(debug_opts);
    
    // Create output video writer
    cv::VideoWriter writer;
    writer.open("tracking_output.avi", 
               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
               fps, cv::Size(frame_width * 2, frame_height));
    
    if (!writer.isOpened()) {
        std::cerr << "Warning: Could not create output video file" << std::endl;
    }
    
    cv::Mat prev_frame, prev_gray;
    std::vector<cv::Point2f> prev_points;
    bool first_frame = true;
    int frame_count = 0;
    
    std::cout << "\nPress SPACE to pause, ESC to quit" << std::endl;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) break;
        
        frame_count++;
        std::cout << "\rProcessing frame " << frame_count << "/" << total_frames << std::flush;
        
        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame;
        }
        
        if (first_frame) {
            // Extract features from first frame
            auto features = extractor->extract(frame);
            
            // Convert keypoints to points
            prev_points.reserve(features.keypoints.size());
            for (const auto& kp : features.keypoints) {
                prev_points.push_back(kp.pt);
            }
            
            prev_frame = frame.clone();
            prev_gray = gray.clone();
            first_frame = false;
            
            // Draw features on first frame
            cv::Mat display = frame.clone();
            for (const auto& pt : prev_points) {
                cv::circle(display, pt, 3, cv::Scalar(0, 255, 0), -1);
            }
            cv::putText(display, "Initial Features: " + std::to_string(prev_points.size()),
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                       cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("Feature Tracking", display);
            if (writer.isOpened()) {
                writer.write(display);
            }
            
            int key = cv::waitKey(1);
            if (key == 27) break;  // ESC
            if (key == 32) {       // SPACE
                key = cv::waitKey(0);
                if (key == 27) break;
            }
            
            continue;
        }
        
        // Track features
        auto result = tracker->track(prev_frame, frame, prev_points);
        
        // Create visualization
        cv::Mat display;
        cv::hconcat(prev_frame, frame, display);
        
        // Draw tracking results
        int tracked_count = 0;
        for (size_t i = 0; i < result.prev_points.size(); ++i) {
            if (result.status[i]) {
                tracked_count++;
                
                // Color based on tracking error
                double error = result.errors[i];
                cv::Scalar color;
                if (error < 1.0) color = cv::Scalar(0, 255, 0);      // Green
                else if (error < 5.0) color = cv::Scalar(0, 255, 255); // Yellow
                else color = cv::Scalar(0, 0, 255);                    // Red
                
                cv::Point2f prev_pt = result.prev_points[i];
                cv::Point2f curr_pt = result.current_points[i] + 
                                    cv::Point2f(prev_frame.cols, 0);
                
                cv::line(display, prev_pt, curr_pt, color, 1);
                cv::circle(display, prev_pt, 2, color, -1);
                cv::circle(display, curr_pt, 2, color, -1);
                
                // Draw track ID
                cv::putText(display, std::to_string(result.track_ids[i]),
                           curr_pt + cv::Point2f(5, -5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
            }
        }
        
        // Draw statistics
        std::ostringstream ss;
        ss << "Frame: " << frame_count << "/" << total_frames 
           << " | Tracked: " << tracked_count << "/" << result.prev_points.size()
           << " | Avg Error: " << std::fixed << std::setprecision(2) << result.avg_error;
        std::string stats = ss.str();
        cv::putText(display, stats, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Feature Tracking", display);
        if (writer.isOpened()) {
            writer.write(display);
        }
        
        // Update for next frame
        prev_points.clear();
        for (size_t i = 0; i < result.current_points.size(); ++i) {
            if (result.status[i]) {
                prev_points.push_back(result.current_points[i]);
            }
        }
        
        prev_frame = frame.clone();
        prev_gray = gray.clone();
        
        // Keyboard controls
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC
        if (key == 32) {       // SPACE
            key = cv::waitKey(0);
            if (key == 27) break;
        }
        
        // If too few points tracked, re-detect
        if (prev_points.size() < 30) {
            std::cout << "\nToo few points tracked (" << prev_points.size() 
                      << "), re-detecting..." << std::endl;
            
            auto features = extractor->extract(frame);
            prev_points.clear();
            for (const auto& kp : features.keypoints) {
                prev_points.push_back(kp.pt);
            }
        }
    }
    
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    std::cout << "\nProcessing complete!" << std::endl;
    std::cout << "Debug files saved to: " << debug_opts.output_dir << std::endl;
    if (writer.isOpened()) {
        std::cout << "Output video saved as: tracking_output.avi" << std::endl;
    }
    
    return 0;
}