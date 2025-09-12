#pragma once
#include "slam/modules.hpp"
#include <ceres/ceres.h>

namespace slam {

// Standard triangulation implementation
class StandardTriangulator : public DepthEstimator {
public:
    StandardTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // Additional triangulation options
    void setMethod(int method);  // CV_TRIANGULATE_*
    void setMinDisparity(double min_disparity);
    void setAngleCheck(bool enable);
    
private:
    int triangulation_method_;
    double min_disparity_;
    bool enable_angle_check_;
    
    double calculateParallaxAngle(const Eigen::Vector3d& ray1, 
                                const Eigen::Vector3d& ray2);
};

// Mid-point method for triangulation
class MidpointTriangulator : public DepthEstimator {
public:
    MidpointTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // Iterative refinement options
    void setMaxIterations(int max_iter);
    void setConvergenceThreshold(double threshold);
    void setUseWeights(bool use_weights);
    
private:
    int max_iterations_;
    double convergence_threshold_;
    bool use_weights_;
    
    Eigen::Vector3d midpointTriangulation(
        const Eigen::Vector3d& ray1,
        const Eigen::Vector3d& ray2,
        const Eigen::Matrix4d& pose1,
        const Eigen::Matrix4d& pose2);
};

// DLT (Direct Linear Transform) triangulation
class DLTTriangulator : public DepthEstimator {
public:
    DLTTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // DLT options
    void setNormalizePoints(bool normalize);
    void setUseRansac(bool use_ransac);
    void setRansacThreshold(double threshold);
    
private:
    bool normalize_points_;
    bool use_ransac_;
    double ransac_threshold_;
    
    cv::Mat normalizePoints(const std::vector<cv::Point2f>& points,
                           std::vector<cv::Point2f>& normalized_points,
                           cv::Mat& T);
    
    Eigen::Vector3d dltTriangulation(
        const cv::Point2f& pt1,
        const cv::Point2f& pt2,
        const Eigen::Matrix3d& K1,
        const Eigen::Matrix3d& K2,
        const Eigen::Matrix4d& pose1,
        const Eigen::Matrix4d& pose2);
    
    friend class OptimalTriangulator;
};

// Optimal triangulation (minimizing reprojection error)
class OptimalTriangulator : public DepthEstimator {
public:
    OptimalTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // Optimization options
    void setUseCeres(bool use_ceres);
    void setMaxIterations(int max_iter);
    void setHuberThreshold(double threshold);
    
private:
    bool use_ceres_;
    int max_iterations_;
    double huber_threshold_;
    double convergence_threshold_;
    
    // Ceres cost function for triangulation
    struct TriangulationCostFunctor {
        TriangulationCostFunctor(const cv::Point2f& pt1, 
                               const cv::Point2f& pt2,
                               const Eigen::Matrix3d& K1,
                               const Eigen::Matrix3d& K2,
                               const Eigen::Matrix4d& pose1,
                               const Eigen::Matrix4d& pose2)
            : pt1_(pt1), pt2_(pt2), K1_(K1), K2_(K2), pose1_(pose1), pose2_(pose2) {}
        
        template <typename T>
        bool operator()(const T* const point, T* residual) const;
        
    private:
        cv::Point2f pt1_, pt2_;
        Eigen::Matrix3d K1_, K2_;
        Eigen::Matrix4d pose1_, pose2_;
    };
    
    Eigen::Vector3d optimalTriangulationCeres(
        const cv::Point2f& pt1,
        const cv::Point2f& pt2,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix4d& pose1,
        const Eigen::Matrix4d& pose2);
    
    Eigen::Vector3d optimalTriangulationGaussNewton(
        const cv::Point2f& pt1,
        const cv::Point2f& pt2,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix4d& pose1,
        const Eigen::Matrix4d& pose2);
};

// Multi-frame triangulation for improved accuracy
class MultiFrameTriangulator : public DepthEstimator {
public:
    MultiFrameTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // Add more observations for a point
    void addObservation(const cv::Point2f& point, const Eigen::Matrix4d& pose);
    void clearObservations();
    
    // Multi-frame options
    void setMinObservations(int min_obs);
    void setOutlierThreshold(double threshold);
    
private:
    struct Observation {
        cv::Point2f point;
        Eigen::Matrix4d pose;
    };
    
    std::vector<Observation> observations_;
    int min_observations_;
    double outlier_threshold_;
    
    std::vector<bool> outlierRejection(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Observation>& obs);
};

// Patch-based stereo triangulation
class PatchStereoTriangulator : public DepthEstimator {
public:
    PatchStereoTriangulator();
    
    DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                          const std::vector<cv::Point2f>& points2,
                          const Eigen::Matrix4d& pose1,
                          const Eigen::Matrix4d& pose2,
                          const cv::Mat& camera_matrix,
                          const cv::Mat& dist_coeffs,
                          const cv::Mat& image1 = cv::Mat(),
                          const cv::Mat& image2 = cv::Mat()) override;
    
    // Patch matching options
    void setPatchSize(int size);
    void setDisparityRange(int min_disp, int max_disp);
    void setSubpixelRefinement(bool enable);
    void setUniquenessRatio(double ratio);
    
private:
    int patch_size_;
    int min_disparity_;
    int max_disparity_;
    bool subpixel_refinement_;
    double uniqueness_ratio_;
    
    double computePatchMatch(const cv::Mat& patch1, const cv::Mat& patch2);
    cv::Point2f subpixelRefinement(const cv::Mat& image, const cv::Point2f& point);
};

// Factory function to create depth estimators
std::unique_ptr<DepthEstimator> createDepthEstimator(
    const std::string& type = "Standard",
    const DepthEstimator::Options& options = DepthEstimator::Options()
);

} // namespace slam