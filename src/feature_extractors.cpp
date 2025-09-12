#include "slam/feature_extractors.hpp"
#include <chrono>

// Use type aliases for cleaner code
using Features = slam::FeatureExtractor::Features;

namespace slam {

// ORB Feature Extractor Implementation
ORBFeatureExtractor::ORBFeatureExtractor() {
    // Initialize ORB with default parameters
    orb_ = cv::ORB::create(options_.max_features);
}

Features ORBFeatureExtractor::extract(const cv::Mat& image) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Features features;
    features.image = image.clone();
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    
    // Apply mask if provided
    cv::Mat masked_image = gray_image;
    if (!features.mask.empty()) {
        gray_image.copyTo(masked_image, features.mask);
    }
    
    // Detect and compute
    orb_->detectAndCompute(masked_image, cv::noArray(), features.keypoints, features.descriptors);
    
    // Filter keypoints based on response
    if (options_.quality_level > 0) {
        std::sort(features.keypoints.begin(), features.keypoints.end(),
                 [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                     return a.response > b.response;
                 });
        
        int n_features = std::min(static_cast<int>(features.keypoints.size()), 
                                 options_.max_features);
        features.keypoints.resize(n_features);
        
        // Update descriptors accordingly
        cv::Mat new_descriptors(n_features, features.descriptors.cols, 
                               features.descriptors.type());
        for (int i = 0; i < n_features; ++i) {
            features.descriptors.row(i).copyTo(new_descriptors.row(i));
        }
        features.descriptors = new_descriptors;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    features.extraction_time_ms = duration.count();
    
    visualize(features);
    
    return features;
}

void ORBFeatureExtractor::setScaleFactor(double factor) {
    orb_->setScaleFactor(factor);
}

void ORBFeatureExtractor::setNLevels(int levels) {
    orb_->setNLevels(levels);
}

void ORBFeatureExtractor::setEdgeThreshold(int threshold) {
    orb_->setEdgeThreshold(threshold);
}

void ORBFeatureExtractor::setFirstLevel(int level) {
    orb_->setFirstLevel(level);
}

void ORBFeatureExtractor::setWTA_K(int k) {
    orb_->setWTA_K(k);
}

void ORBFeatureExtractor::setScoreType(int type) {
    orb_->setScoreType(static_cast<cv::ORB::ScoreType>(type));
}

void ORBFeatureExtractor::setPatchSize(int size) {
    orb_->setPatchSize(size);
}

void ORBFeatureExtractor::setFastThreshold(int threshold) {
    orb_->setFastThreshold(threshold);
}

// Shi-Tomasi Feature Extractor Implementation
ShiTomasiFeatureExtractor::ShiTomasiFeatureExtractor() 
    : max_corners_(1000), quality_level_(0.01), min_distance_(10.0),
      block_size_(3), use_harris_(false), k_(0.04) {
}

Features ShiTomasiFeatureExtractor::extract(const cv::Mat& image) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Features features;
    features.image = image.clone();
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    
    // Detect corners
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray_image, corners, max_corners_, quality_level_,
                           min_distance_, features.mask, block_size_, 
                           use_harris_, k_);
    
    // Convert to KeyPoint format
    features.keypoints.reserve(corners.size());
    for (const auto& corner : corners) {
        cv::KeyPoint kp(corner, options_.block_size * 2);
        features.keypoints.push_back(kp);
    }
    
    // Since Shi-Tomasi doesn't provide descriptors, we'll use ORB descriptors
    // at the detected corner locations
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(gray_image, features.keypoints, features.descriptors);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    features.extraction_time_ms = duration.count();
    
    visualize(features);
    
    return features;
}

void ShiTomasiFeatureExtractor::setMaxCorners(int max_corners) {
    max_corners_ = max_corners;
}

void ShiTomasiFeatureExtractor::setQualityLevel(double quality_level) {
    quality_level_ = quality_level;
}

void ShiTomasiFeatureExtractor::setMinDistance(double min_distance) {
    min_distance_ = min_distance;
}

void ShiTomasiFeatureExtractor::setBlockSize(int block_size) {
    block_size_ = block_size;
}

void ShiTomasiFeatureExtractor::setUseHarrisDetector(bool use_harris) {
    use_harris_ = use_harris;
}

void ShiTomasiFeatureExtractor::setK(double k) {
    k_ = k;
}

// FAST+ORB Feature Extractor Implementation
FASTORBFeatureExtractor::FASTORBFeatureExtractor() {
    fast_ = cv::FastFeatureDetector::create(10);
    orb_ = cv::ORB::create(500);
}

Features FASTORBFeatureExtractor::extract(const cv::Mat& image) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Features features;
    features.image = image.clone();
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    
    // Detect FAST corners
    fast_->detect(gray_image, features.keypoints);
    
    // Filter keypoints based on response
    std::sort(features.keypoints.begin(), features.keypoints.end(),
             [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                 return a.response > b.response;
             });
    
    int n_features = std::min(static_cast<int>(features.keypoints.size()), 
                             options_.max_features);
    features.keypoints.resize(n_features);
    
    // Compute ORB descriptors
    orb_->compute(gray_image, features.keypoints, features.descriptors);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    features.extraction_time_ms = duration.count();
    
    visualize(features);
    
    return features;
}

void FASTORBFeatureExtractor::setFASTThreshold(int threshold) {
    fast_->setThreshold(threshold);
}

void FASTORBFeatureExtractor::setFASTType(int type) {
    fast_->setType(static_cast<cv::FastFeatureDetector::DetectorType>(type));
}

void FASTORBFeatureExtractor::setORBScaleFactor(double factor) {
    orb_->setScaleFactor(factor);
}

void FASTORBFeatureExtractor::setORBLevels(int levels) {
    orb_->setNLevels(levels);
}

// Factory function implementation
std::unique_ptr<FeatureExtractor> createFeatureExtractor(
    const std::string& type,
    const FeatureExtractor::Options& options) {
    
    std::unique_ptr<FeatureExtractor> extractor;
    
    if (type == "ORB") {
        extractor = std::make_unique<ORBFeatureExtractor>();
    } else if (type == "ShiTomasi") {
        extractor = std::make_unique<ShiTomasiFeatureExtractor>();
    } else if (type == "FAST_ORB") {
        extractor = std::make_unique<FASTORBFeatureExtractor>();
    } else {
        // Default to ORB
        extractor = std::make_unique<ORBFeatureExtractor>();
    }
    
    extractor->setOptions(options);
    return extractor;
}

} // namespace slam