#pragma once
#include "slam/modules.hpp"
#include <opencv2/video/tracking.hpp>

namespace slam {

// LK optical flow tracker implementation
class LKOpticalFlowTracker : public FeatureTracker {
public:
    LKOpticalFlowTracker();
    
    TrackingResult track(const cv::Mat& prev_img, 
                        const cv::Mat& curr_img,
                        const std::vector<cv::Point2f>& prev_points) override;
    
    // Additional LK-specific options
    void setFlags(int flags);
    void setMinEigThreshold(double threshold);
    void setMaxLevel(int max_level);
    
private:
    std::vector<cv::Point2f> prev_points_pyr_[2];
    std::vector<int> track_ids_;
    int next_track_id_;
};

// Feature matching tracker implementation
class FeatureMatchingTracker : public FeatureTracker {
public:
    FeatureMatchingTracker();
    
    TrackingResult track(const cv::Mat& prev_img, 
                        const cv::Mat& curr_img,
                        const std::vector<cv::Point2f>& prev_points) override;
    
    // Set feature extractor for matching
    void setFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor);
    
    // Matching options
    void setMatcherType(const std::string& type);
    void setMatchRatio(double ratio);
    void setCrossCheck(bool enable);
    void setUseOpticalFlowRefinement(bool enable);
    
private:
    std::shared_ptr<FeatureExtractor> feature_extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    double match_ratio_;
    bool cross_check_;
    bool use_optical_flow_refinement_;
    FeatureExtractor::Features prev_features_;
    
    std::vector<cv::DMatch> filterMatches(const std::vector<cv::DMatch>& matches);
};

// Pyramidal LK optical flow with backward tracking
class PyramidLKTracker : public FeatureTracker {
public:
    PyramidLKTracker();
    
    TrackingResult track(const cv::Mat& prev_img, 
                        const cv::Mat& curr_img,
                        const std::vector<cv::Point2f>& prev_points) override;
    
    // Backward tracking for consistency check
    void enableBackwardTracking(bool enable);
    void setBackwardThreshold(double threshold);
    
private:
    bool use_backward_tracking_;
    double backward_threshold_;
    std::vector<int> track_ids_;
    int next_track_id_;
    
    std::vector<uchar> backwardTrackingConsistencyCheck(
        const cv::Mat& prev_img,
        const cv::Mat& curr_img,
        const std::vector<cv::Point2f>& prev_points,
        const std::vector<cv::Point2f>& forward_points,
        const std::vector<uchar>& forward_status
    );
};

// Sparse optical flow with outlier rejection
class SparseFlowTracker : public FeatureTracker {
public:
    SparseFlowTracker();
    
    TrackingResult track(const cv::Mat& prev_img, 
                        const cv::Mat& curr_img,
                        const std::vector<cv::Point2f>& prev_points) override;
    
    // Outlier rejection options
    void setRansacThreshold(double threshold);
    void setMinInlierRatio(double ratio);
    void setEnableFundamentalCheck(bool enable);
    void setEnableHomographyCheck(bool enable);
    
private:
    double ransac_threshold_;
    double min_inlier_ratio_;
    bool enable_fundamental_check_;
    bool enable_homography_check_;
    std::vector<int> track_ids_;
    int next_track_id_;
    
    std::vector<uchar> geometricConsistencyCheck(
        const cv::Mat& prev_img,
        const cv::Mat& curr_img,
        const std::vector<cv::Point2f>& prev_points,
        const std::vector<cv::Point2f>& curr_points
    );
};

// Deep learning based tracker (placeholder for future implementation)
class DeepFeatureTracker : public FeatureTracker {
public:
    DeepFeatureTracker();
    
    TrackingResult track(const cv::Mat& prev_img, 
                        const cv::Mat& curr_img,
                        const std::vector<cv::Point2f>& prev_points) override;
    
    // Model options
    void setModelPath(const std::string& path);
    void setConfidenceThreshold(float threshold);
    
private:
    std::string model_path_;
    float confidence_threshold_;
};

// Factory function to create feature trackers
std::unique_ptr<FeatureTracker> createFeatureTracker(
    const std::string& type = "LK",
    const FeatureTracker::Options& options = FeatureTracker::Options()
);

} // namespace slam