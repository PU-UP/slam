#pragma once
#include "slam/modules.hpp"
#include <ceres/ceres.h>

namespace slam {

// Standard Ceres-based bundle adjustment
class CeresBundleAdjuster : public BundleAdjuster {
public:
    CeresBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Ceres-specific options
    void setLossFunction(const std::string& type);
    void setLinearSolverType(const std::string& type);
    void setUpdateType(const std::string& type);
    void setTrustRegionStrategy(const std::string& type);
    
    // Set custom callbacks
    void setIterationCallback(std::function<void(const ceres::IterationSummary&)> callback);
    
private:
    struct ReprojectionError {
        ReprojectionError(double observed_x, double observed_y,
                        double focal_x, double focal_y,
                        double principal_x, double principal_y)
            : observed_x_(observed_x), observed_y_(observed_y),
              focal_x_(focal_x), focal_y_(focal_y),
              principal_x_(principal_x), principal_y_(principal_y) {}
        
        template <typename T>
        bool operator()(const T* const camera_pose, const T* const point, T* residuals) const;
        
        double observed_x_, observed_y_;
        double focal_x_, focal_y_;
        double principal_x_, principal_y_;
    };
    
    struct ReprojectionErrorWithDistortion {
        ReprojectionErrorWithDistortion(double observed_x, double observed_y,
                                       const double* camera_params)
            : observed_x_(observed_x), observed_y_(observed_y),
              camera_params_(camera_params) {}
        
        template <typename T>
        bool operator()(const T* const camera_pose, const T* const point, T* residuals) const;
        
        double observed_x_, observed_y_;
        const double* camera_params_;
    };
    
    struct WheelOdometryPrior {
        WheelOdometryPrior(const double* pose1, const double* pose2,
                          double trans_weight, double rot_weight)
            : pose1_(pose1), pose2_(pose2),
              trans_weight_(trans_weight), rot_weight_(rot_weight) {}
        
        template <typename T>
        bool operator()(const T* const pose1_opt, const T* const pose2_opt, T* residuals) const;
        
        const double* pose1_;
        const double* pose2_;
        double trans_weight_;
        double rot_weight_;
    };
    
    std::function<void(const ceres::IterationSummary&)> iteration_callback_;
};

// Robust bundle adjustment with automatic outlier rejection
class RobustBundleAdjuster : public BundleAdjuster {
public:
    RobustBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Robust estimation options
    void setOutlierRejectionMethod(const std::string& method);
    void setOutlierThreshold(double threshold);
    void setMaxOutlierIterations(int max_iter);
    void setMinInlierRatio(double ratio);
    
private:
    std::string outlier_method_;
    double outlier_threshold_;
    int max_outlier_iter_;
    double min_inlier_ratio_;
    
    std::vector<bool> detectOutliers(const BAProblem& problem,
                                    const std::vector<double>& residuals);
    
    void removeOutliers(BAProblem& problem, const std::vector<bool>& is_inlier);
};

// Incremental bundle adjustment for real-time applications
class IncrementalBundleAdjuster : public BundleAdjuster {
public:
    IncrementalBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Incremental BA options
    void setWindowSize(int window_size);
    void setKeyframeStrategy(const std::string& strategy);
    void setMarginalization(bool enable);
    void setUpdateFrequency(int frequency);
    
    // Add new observations incrementally
    void addKeyframe(const Eigen::Matrix4d& pose,
                    const std::vector<int>& point_indices,
                    const std::vector<cv::Point2f>& observations);
    void addObservations(int frame_idx, 
                        const std::vector<int>& point_indices,
                        const std::vector<cv::Point2f>& observations);
    
private:
    struct FrameState {
        Eigen::Matrix4d pose;
        bool is_keyframe;
        double timestamp;
        std::vector<int> observed_points;
        std::vector<cv::Point2f> observations;
    };
    
    struct PointState {
        Eigen::Vector3d position;
        int observations_count;
        double quality_score;
        bool is_fixed;
    };
    
    int window_size_;
    std::string keyframe_strategy_;
    bool enable_marginalization_;
    int update_frequency_;
    
    std::vector<FrameState> frames_;
    std::vector<PointState> points_;
    int last_optimized_frame_;
    
    bool shouldInsertKeyframe(const FrameState& frame);
    void marginalizeOldFrames();
    void selectActiveWindow(BAProblem& problem);
};

// Pose-graph bundle adjustment for loop closure
class PoseGraphBundleAdjuster : public BundleAdjuster {
public:
    PoseGraphBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Pose graph options
    void addLoopClosure(int frame_idx1, int frame_idx2, 
                       const Eigen::Matrix4d& relative_pose,
                       double information);
    void addRelativePoseConstraint(int frame_idx1, int frame_idx2,
                                  const Eigen::Matrix4d& relative_pose,
                                  double information);
    
    // Loop closure options
    void setLoopClosureThreshold(double threshold);
    void setLoopClosureVerification(bool enable);
    
private:
    struct LoopClosure {
        int frame1_idx;
        int frame2_idx;
        Eigen::Matrix4d relative_pose;
        double information_matrix[6][6];
        bool verified;
    };
    
    std::vector<LoopClosure> loop_closures_;
    double loop_threshold_;
    bool verify_loop_closures_;
    
    bool detectLoopClosures(const BAProblem& problem);
    bool verifyLoopClosure(const LoopClosure& closure, const BAProblem& problem);
    void buildPoseGraph(const BAProblem& problem);
};

// Distributed bundle adjustment for large-scale problems
class DistributedBundleAdjuster : public BundleAdjuster {
public:
    DistributedBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Distributed options
    void setNumSubmaps(int num_submaps);
    void setCommunicationStrategy(const std::string& strategy);
    void setOverlapSize(int overlap);
    
private:
    int num_submaps_;
    std::string comm_strategy_;
    int overlap_size_;
    
    struct Submap {
        std::vector<int> frame_indices;
        std::vector<int> point_indices;
        std::vector<std::pair<int, int>> observation_indices;
        Eigen::Vector3d center;
        double radius;
    };
    
    std::vector<Submap> partitionProblem(const BAProblem& problem);
    void optimizeSubmaps(const std::vector<Submap>& submaps,
                        const BAProblem& problem);
    void mergeSubmaps(const std::vector<Submap>& submaps,
                     std::vector<Eigen::Matrix4d>& optimized_poses,
                     std::vector<Eigen::Vector3d>& optimized_points);
};

// GPU-accelerated bundle adjustment
class GPUBundleAdjuster : public BundleAdjuster {
public:
    GPUBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // GPU options
    void setDeviceId(int device_id);
    void setUseGPU(bool use_gpu);
    void setBlockSize(int block_size);
    
private:
    int device_id_;
    bool use_gpu_;
    int block_size_;
    
    bool checkGPUSupport();
    void setupGPUData(const BAProblem& problem);
    void optimizeOnGPU(BAProblem& problem);
};

// Lightweight bundle adjustment for embedded systems
class LightweightBundleAdjuster : public BundleAdjuster {
public:
    LightweightBundleAdjuster();
    
    BAResult optimize(BAProblem& problem) override;
    
    // Lightweight options
    void setUseSchurComplement(bool enable);
    void setUseParameterBlockOrdering(bool enable);
    void setUseReducedCamera(bool enable);
    
private:
    bool use_schur_complement_;
    bool use_parameter_ordering_;
    bool use_reduced_camera_;
    
    void setupSchurComplement(ceres::Problem& problem, BAProblem& ba_problem);
    void setupParameterOrdering(ceres::Problem& problem, BAProblem& ba_problem);
};

// Factory function to create bundle adjusters
std::unique_ptr<BundleAdjuster> createBundleAdjuster(
    const std::string& type = "Ceres",
    const BundleAdjuster::Options& options = BundleAdjuster::Options()
);

} // namespace slam