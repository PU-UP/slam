#pragma once
#include "slam/modules.hpp"
#include <opencv2/features2d.hpp>

namespace slam {

// ORB feature extractor implementation
class ORBFeatureExtractor : public FeatureExtractor {
public:
    ORBFeatureExtractor();
    
    Features extract(const cv::Mat& image) override;
    
    // Additional ORB-specific options
    void setScaleFactor(double factor);
    void setNLevels(int levels);
    void setEdgeThreshold(int threshold);
    void setFirstLevel(int level);
    void setWTA_K(int k);
    void setScoreType(int type);
    void setPatchSize(int size);
    void setFastThreshold(int threshold);
    
private:
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

// Shi-Tomasi corner detector implementation
class ShiTomasiFeatureExtractor : public FeatureExtractor {
public:
    ShiTomasiFeatureExtractor();
    
    Features extract(const cv::Mat& image) override;
    
    // Additional options
    void setMaxCorners(int max_corners);
    void setQualityLevel(double quality_level);
    void setMinDistance(double min_distance);
    void setBlockSize(int block_size);
    void setUseHarrisDetector(bool use_harris);
    void setK(double k);
    
private:
    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    bool use_harris_;
    double k_;
};

// Note: SIFT and SURF require OpenCV contrib modules
// They have been temporarily removed to avoid external dependencies

// FAST corner detector with ORB descriptors
class FASTORBFeatureExtractor : public FeatureExtractor {
public:
    FASTORBFeatureExtractor();
    
    Features extract(const cv::Mat& image) override;
    
    // Additional options
    void setFASTThreshold(int threshold);
    void setFASTType(int type);
    void setORBScaleFactor(double factor);
    void setORBLevels(int levels);
    
private:
    cv::Ptr<cv::FastFeatureDetector> fast_;
    cv::Ptr<cv::ORB> orb_;
};

// Factory function to create feature extractors
std::unique_ptr<FeatureExtractor> createFeatureExtractor(
    const std::string& type = "ORB",
    const FeatureExtractor::Options& options = FeatureExtractor::Options()
);

} // namespace slam