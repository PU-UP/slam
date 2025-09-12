#include "slam/feature_trackers.hpp"
#include "slam/feature_extractors.hpp"
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <numeric>

// Use type aliases for cleaner code
using TrackingResult = slam::FeatureTracker::TrackingResult;
using Features = slam::FeatureExtractor::Features;

namespace slam {

// LK Optical Flow Tracker Implementation
LKOpticalFlowTracker::LKOpticalFlowTracker() : next_track_id_(0) {
    // Initialize with default options
    options_.max_level = 3;
    options_.win_size = cv::Size(21, 21);
    options_.criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
}

TrackingResult LKOpticalFlowTracker::track(const cv::Mat& prev_img, 
                                          const cv::Mat& curr_img,
                                          const std::vector<cv::Point2f>& prev_points) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TrackingResult result;
    result.prev_image = prev_img.clone();
    result.current_image = curr_img.clone();
    
    if (prev_points.empty()) {
        result.tracking_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to grayscale if needed
    cv::Mat prev_gray, curr_gray;
    if (prev_img.channels() == 3) {
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = prev_img;
    }
    
    if (curr_img.channels() == 3) {
        cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
    } else {
        curr_gray = curr_img;
    }
    
    // Initialize track IDs if this is the first frame
    if (track_ids_.empty()) {
        track_ids_.resize(prev_points.size());
        for (size_t i = 0; i < prev_points.size(); ++i) {
            track_ids_[i] = next_track_id_++;
        }
    }
    
    // Track points
    std::vector<float> err;
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, 
                           result.current_points, status, err,
                           options_.win_size, options_.max_level,
                           options_.criteria, 0, 1e-4);
    
    // Filter out points with high error
    result.prev_points = prev_points;
    result.status = status;
    result.errors = err;
    result.track_ids = track_ids_;
    
    // Calculate statistics
    int tracked_count = 0;
    double total_error = 0.0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && err[i] < options_.max_error) {
            tracked_count++;
            total_error += err[i];
        } else {
            status[i] = 0;  // Mark as lost
        }
    }
    
    result.tracked_count = tracked_count;
    result.avg_error = tracked_count > 0 ? total_error / tracked_count : 0.0;
    
    // Update track IDs for next frame
    std::vector<int> new_track_ids;
    std::vector<cv::Point2f> new_prev_points;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            new_track_ids.push_back(track_ids_[i]);
            new_prev_points.push_back(result.current_points[i]);
        }
    }
    track_ids_ = new_track_ids;
    prev_points_pyr_[0] = new_prev_points;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.tracking_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

void LKOpticalFlowTracker::setFlags(int /*flags*/) {
    // Additional flags for calcOpticalFlowPyrLK
}

void LKOpticalFlowTracker::setMinEigThreshold(double threshold) {
    options_.min_eigen_threshold = threshold;
}

void LKOpticalFlowTracker::setMaxLevel(int max_level) {
    options_.max_level = max_level;
}

// Feature Matching Tracker Implementation
FeatureMatchingTracker::FeatureMatchingTracker() 
    : match_ratio_(0.75), cross_check_(true), use_optical_flow_refinement_(true) {
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

TrackingResult FeatureMatchingTracker::track(const cv::Mat& prev_img, 
                                            const cv::Mat& curr_img,
                                            const std::vector<cv::Point2f>& prev_points) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TrackingResult result;
    result.prev_image = prev_img.clone();
    result.current_image = curr_img.clone();
    
    if (prev_points.empty()) {
        result.tracking_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Extract features from current frame
    auto curr_features = feature_extractor_->extract(curr_img);
    
    if (prev_features_.descriptors.empty()) {
        // First frame, just extract features
        prev_features_ = feature_extractor_->extract(prev_img);
        
        // Find matches with provided prev_points
        for (const auto& kp : prev_features_.keypoints) {
            // Find closest point in prev_points
            float min_dist = std::numeric_limits<float>::max();
            int closest_idx = -1;
            
            for (size_t i = 0; i < prev_points.size(); ++i) {
                float dist = cv::norm(kp.pt - prev_points[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_idx = i;
                }
            }
            
            if (min_dist < 5.0f) {  // Threshold for matching
                result.prev_points.push_back(prev_points[closest_idx]);
                result.track_ids.push_back(closest_idx);
            }
        }
        
        result.tracking_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Match features between frames
    std::vector<cv::DMatch> matches;
    if (cross_check_) {
        std::vector<cv::DMatch> matches12, matches21;
        matcher_->match(prev_features_.descriptors, curr_features.descriptors, matches12);
        matcher_->match(curr_features.descriptors, prev_features_.descriptors, matches21);
        
        // Cross-check
        for (const auto& m12 : matches12) {
            for (const auto& m21 : matches21) {
                if (m12.queryIdx == m21.trainIdx && m12.trainIdx == m21.queryIdx) {
                    matches.push_back(m12);
                    break;
                }
            }
        }
    } else {
        matcher_->match(prev_features_.descriptors, curr_features.descriptors, matches);
    }
    
    // Filter matches by ratio test if applicable
    if (match_ratio_ > 0) {
        std::vector<cv::DMatch> good_matches = filterMatches(matches);
        matches = good_matches;
    }
    
    // Convert matches to tracking result
    result.prev_points.reserve(matches.size());
    result.current_points.reserve(matches.size());
    result.track_ids.reserve(matches.size());
    result.status.resize(matches.size(), 1);
    result.errors.resize(matches.size(), 0.0f);
    
    for (const auto& match : matches) {
        result.prev_points.push_back(prev_features_.keypoints[match.queryIdx].pt);
        result.current_points.push_back(curr_features.keypoints[match.trainIdx].pt);
        result.track_ids.push_back(match.queryIdx);
    }
    
    // Optional optical flow refinement
    if (use_optical_flow_refinement_) {
        cv::Mat prev_gray, curr_gray;
        if (prev_img.channels() == 3) {
            cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        } else {
            prev_gray = prev_img;
        }
        
        if (curr_img.channels() == 3) {
            cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
        } else {
            curr_gray = curr_img;
        }
        
        std::vector<cv::Point2f> refined_points = result.current_points;
        std::vector<float> err;
        std::vector<uchar> status;
        
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, result.prev_points,
                               refined_points, status, err,
                               cv::Size(11, 11), 3,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01));
        
        // Update with refined positions
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                result.current_points[i] = refined_points[i];
                result.errors[i] = err[i];
            } else {
                result.status[i] = 0;
            }
        }
    }
    
    // Calculate statistics
    result.tracked_count = std::count(result.status.begin(), result.status.end(), 1);
    result.avg_error = result.tracked_count > 0 ? 
        std::accumulate(result.errors.begin(), result.errors.end(), 0.0) / result.tracked_count : 0.0;
    
    // Update for next frame
    prev_features_ = curr_features;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.tracking_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

void FeatureMatchingTracker::setFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor) {
    feature_extractor_ = extractor;
}

void FeatureMatchingTracker::setMatcherType(const std::string& type) {
    if (type == "BF_HAMMING") {
        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    } else if (type == "BF_L2") {
        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    } else if (type == "FLANN") {
        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
}

void FeatureMatchingTracker::setMatchRatio(double ratio) {
    match_ratio_ = ratio;
}

void FeatureMatchingTracker::setCrossCheck(bool enable) {
    cross_check_ = enable;
}

void FeatureMatchingTracker::setUseOpticalFlowRefinement(bool enable) {
    use_optical_flow_refinement_ = enable;
}

std::vector<cv::DMatch> FeatureMatchingTracker::filterMatches(const std::vector<cv::DMatch>&) {
    std::vector<cv::DMatch> good_matches;
    
    // Note: This function is no longer needed with the new implementation
    // For ratio test, we need knn matches
    /*std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(prev_features_.descriptors, curr_features.descriptors, knn_matches, 2);
    
    for (const auto& knn_match : knn_matches) {
        if (knn_match.size() >= 2) {
            if (knn_match[0].distance < match_ratio_ * knn_match[1].distance) {
                good_matches.push_back(knn_match[0]);
            }
        }
    }*/
    
    return good_matches;
}

// Pyramid LK Tracker Implementation
PyramidLKTracker::PyramidLKTracker() 
    : use_backward_tracking_(true), backward_threshold_(1.0), next_track_id_(0) {
}

TrackingResult PyramidLKTracker::track(const cv::Mat& prev_img, 
                                      const cv::Mat& curr_img,
                                      const std::vector<cv::Point2f>& prev_points) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TrackingResult result;
    result.prev_image = prev_img.clone();
    result.current_image = curr_img.clone();
    
    if (prev_points.empty()) {
        result.tracking_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to grayscale
    cv::Mat prev_gray, curr_gray;
    if (prev_img.channels() == 3) {
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = prev_img;
    }
    
    if (curr_img.channels() == 3) {
        cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
    } else {
        curr_gray = curr_img;
    }
    
    // Initialize track IDs
    if (track_ids_.empty()) {
        track_ids_.resize(prev_points.size());
        for (size_t i = 0; i < prev_points.size(); ++i) {
            track_ids_[i] = next_track_id_++;
        }
    }
    
    // Forward tracking
    std::vector<float> err;
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points,
                           result.current_points, status, err,
                           options_.win_size, options_.max_level,
                           options_.criteria);
    
    // Backward tracking if enabled
    if (use_backward_tracking_) {
        std::vector<uchar> backward_status = backwardTrackingConsistencyCheck(
            prev_gray, curr_gray, prev_points, result.current_points, status);
        
        // Combine forward and backward status
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] && !backward_status[i]) {
                status[i] = 0;  // Inconsistent tracking
            }
        }
    }
    
    // Set up result
    result.prev_points = prev_points;
    result.status = status;
    result.errors = err;
    result.track_ids = track_ids_;
    
    // Calculate statistics
    int tracked_count = 0;
    double total_error = 0.0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && err[i] < options_.max_error) {
            tracked_count++;
            total_error += err[i];
        } else {
            status[i] = 0;
        }
    }
    
    result.tracked_count = tracked_count;
    result.avg_error = tracked_count > 0 ? total_error / tracked_count : 0.0;
    
    // Update track IDs for next frame
    std::vector<int> new_track_ids;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            new_track_ids.push_back(track_ids_[i]);
        }
    }
    track_ids_ = new_track_ids;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.tracking_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

void PyramidLKTracker::enableBackwardTracking(bool enable) {
    use_backward_tracking_ = enable;
}

void PyramidLKTracker::setBackwardThreshold(double threshold) {
    backward_threshold_ = threshold;
}

std::vector<uchar> PyramidLKTracker::backwardTrackingConsistencyCheck(
    const cv::Mat& prev_img,
    const cv::Mat& curr_img,
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& forward_points,
    const std::vector<uchar>& forward_status) {
    
    std::vector<cv::Point2f> backward_points;
    std::vector<float> backward_err;
    std::vector<uchar> backward_status;
    
    // Track backward
    cv::calcOpticalFlowPyrLK(curr_img, prev_img, forward_points,
                           backward_points, backward_status, backward_err,
                           options_.win_size, options_.max_level,
                           options_.criteria);
    
    // Check consistency
    std::vector<uchar> consistent_status(forward_status.size(), 0);
    for (size_t i = 0; i < forward_status.size(); ++i) {
        if (forward_status[i] && backward_status[i]) {
            float distance = cv::norm(prev_points[i] - backward_points[i]);
            if (distance < backward_threshold_) {
                consistent_status[i] = 1;
            }
        }
    }
    
    return consistent_status;
}

// Sparse Flow Tracker Implementation
SparseFlowTracker::SparseFlowTracker() 
    : ransac_threshold_(3.0), min_inlier_ratio_(0.5),
      enable_fundamental_check_(true), enable_homography_check_(false),
      next_track_id_(0) {
}

TrackingResult SparseFlowTracker::track(const cv::Mat& prev_img, 
                                      const cv::Mat& curr_img,
                                      const std::vector<cv::Point2f>& prev_points) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TrackingResult result;
    result.prev_image = prev_img.clone();
    result.current_image = curr_img.clone();
    
    if (prev_points.empty()) {
        result.tracking_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to grayscale
    cv::Mat prev_gray, curr_gray;
    if (prev_img.channels() == 3) {
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = prev_img;
    }
    
    if (curr_img.channels() == 3) {
        cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
    } else {
        curr_gray = curr_img;
    }
    
    // Initialize track IDs
    if (track_ids_.empty()) {
        track_ids_.resize(prev_points.size());
        for (size_t i = 0; i < prev_points.size(); ++i) {
            track_ids_[i] = next_track_id_++;
        }
    }
    
    // Initial tracking
    std::vector<float> err;
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points,
                           result.current_points, status, err,
                           options_.win_size, options_.max_level,
                           options_.criteria);
    
    // Geometric consistency check
    if (enable_fundamental_check_ || enable_homography_check_) {
        std::vector<uchar> geom_status = geometricConsistencyCheck(
            prev_gray, curr_gray, prev_points, result.current_points);
        
        // Combine optical flow and geometric status
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] && !geom_status[i]) {
                status[i] = 0;  // Geometric outlier
            }
        }
    }
    
    // Set up result
    result.prev_points = prev_points;
    result.status = status;
    result.errors = err;
    result.track_ids = track_ids_;
    
    // Calculate statistics
    int tracked_count = 0;
    double total_error = 0.0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && err[i] < options_.max_error) {
            tracked_count++;
            total_error += err[i];
        } else {
            status[i] = 0;
        }
    }
    
    result.tracked_count = tracked_count;
    result.avg_error = tracked_count > 0 ? total_error / tracked_count : 0.0;
    
    // Update track IDs for next frame
    std::vector<int> new_track_ids;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            new_track_ids.push_back(track_ids_[i]);
        }
    }
    track_ids_ = new_track_ids;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.tracking_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

void SparseFlowTracker::setRansacThreshold(double threshold) {
    ransac_threshold_ = threshold;
}

void SparseFlowTracker::setMinInlierRatio(double ratio) {
    min_inlier_ratio_ = ratio;
}

void SparseFlowTracker::setEnableFundamentalCheck(bool enable) {
    enable_fundamental_check_ = enable;
}

void SparseFlowTracker::setEnableHomographyCheck(bool enable) {
    enable_homography_check_ = enable;
}

std::vector<uchar> SparseFlowTracker::geometricConsistencyCheck(
    const cv::Mat& /*prev_img*/,
    const cv::Mat& /*curr_img*/,
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& curr_points) {
    
    std::vector<uchar> status(prev_points.size(), 1);
    
    // Prepare point vectors
    std::vector<cv::Point2f> prev_valid, curr_valid;
    std::vector<size_t> indices;
    
    for (size_t i = 0; i < prev_points.size(); ++i) {
        if (prev_points[i].x >= 0 && prev_points[i].y >= 0 &&
            curr_points[i].x >= 0 && curr_points[i].y >= 0) {
            prev_valid.push_back(prev_points[i]);
            curr_valid.push_back(curr_points[i]);
            indices.push_back(i);
        }
    }
    
    if (prev_valid.size() < 8) {
        return std::vector<uchar>(prev_points.size(), 0);
    }
    
    // Fundamental matrix check
    if (enable_fundamental_check_) {
        cv::Mat F = cv::findFundamentalMat(prev_valid, curr_valid, cv::FM_RANSAC, ransac_threshold_);
        
        if (!F.empty()) {
            std::vector<cv::Vec3f> lines;
            cv::computeCorrespondEpilines(curr_valid, 2, F, lines);
            
            for (size_t i = 0; i < lines.size(); ++i) {
                float distance = std::abs(lines[i][0] * prev_valid[i].x + 
                                         lines[i][1] * prev_valid[i].y + 
                                         lines[i][2]) / 
                               std::sqrt(lines[i][0] * lines[i][0] + 
                                        lines[i][1] * lines[i][1]);
                
                if (distance > ransac_threshold_) {
                    status[indices[i]] = 0;
                }
            }
        }
    }
    
    // Homography check (for planar scenes)
    if (enable_homography_check_) {
        cv::Mat H = cv::findHomography(prev_valid, curr_valid, cv::RANSAC, ransac_threshold_);
        
        if (!H.empty()) {
            std::vector<cv::Point2f> projected;
            cv::perspectiveTransform(prev_valid, projected, H);
            
            for (size_t i = 0; i < projected.size(); ++i) {
                float distance = cv::norm(projected[i] - curr_valid[i]);
                if (distance > ransac_threshold_) {
                    status[indices[i]] = 0;
                }
            }
        }
    }
    
    return status;
}

// Deep Feature Tracker Implementation (placeholder)
DeepFeatureTracker::DeepFeatureTracker() : confidence_threshold_(0.5) {
}

TrackingResult DeepFeatureTracker::track(const cv::Mat& prev_img, 
                                        const cv::Mat& curr_img,
                                        const std::vector<cv::Point2f>& prev_points) {
    // Placeholder implementation
    // TODO: Implement deep learning based tracking (e.g., with LightGlue, SuperGlue, etc.)
    
    TrackingResult result;
    result.prev_image = prev_img.clone();
    result.current_image = curr_img.clone();
    
    // For now, just use LK optical flow as baseline
    LKOpticalFlowTracker lk_tracker;
    lk_tracker.setOptions(options_);
    return lk_tracker.track(prev_img, curr_img, prev_points);
}

void DeepFeatureTracker::setModelPath(const std::string& path) {
    model_path_ = path;
}

void DeepFeatureTracker::setConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
}

// Factory function implementation
std::unique_ptr<FeatureTracker> createFeatureTracker(
    const std::string& type,
    const FeatureTracker::Options& options) {
    
    std::unique_ptr<FeatureTracker> tracker;
    
    if (type == "LK" || type == "OpticalFlow") {
        tracker = std::make_unique<LKOpticalFlowTracker>();
    } else if (type == "Matching") {
        tracker = std::make_unique<FeatureMatchingTracker>();
    } else if (type == "PyramidLK") {
        tracker = std::make_unique<PyramidLKTracker>();
    } else if (type == "SparseFlow") {
        tracker = std::make_unique<SparseFlowTracker>();
    } else if (type == "Deep") {
        tracker = std::make_unique<DeepFeatureTracker>();
    } else {
        // Default to LK optical flow
        tracker = std::make_unique<LKOpticalFlowTracker>();
    }
    
    tracker->setOptions(options);
    return tracker;
}

} // namespace slam