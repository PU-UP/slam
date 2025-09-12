#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace slam {

// Forward declarations
class FeatureExtractor;
class FeatureTracker;
class PoseEstimator;
class DepthEstimator;
class BundleAdjuster;

// Base class for all modules with debug visualization
class ModuleBase {
public:
    struct DebugOptions {
        bool enable_visualization = false;
        bool save_intermediate = false;
        std::string output_dir = "./debug";
        std::string module_name = "module";
        
        // Visualization parameters
        int wait_key_delay = 0;
        double scale_factor = 1.0;
        bool show_text = true;
    };
    
    ModuleBase() = default;
    virtual ~ModuleBase() = default;
    
    void setDebugOptions(const DebugOptions& options) { debug_options_ = options; }
    const DebugOptions& getDebugOptions() const { return debug_options_; }
    
    void visualizeImage(const std::string& window_name, const cv::Mat& image) const;
protected:
    DebugOptions debug_options_;
    
    // Helper functions for visualization
    void saveImage(const std::string& filename, const cv::Mat& image) const;
    void saveData(const std::string& filename, const std::string& data) const;
    
    std::string getOutputPath(const std::string& filename) const;
};

// Feature extraction module interface
class FeatureExtractor : public ModuleBase {
public:
    struct Features {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Mat mask;
        cv::Mat image; // Store reference to original image for visualization
        double extraction_time_ms = 0.0;
    };
    
    struct Options {
        int max_features = 2000;
        double quality_level = 0.01;
        double min_distance = 10.0;
        int block_size = 3;
        bool use_harris_detector = false;
        double k = 0.04;
    };
    
    virtual ~FeatureExtractor() = default;
    
    virtual Features extract(const cv::Mat& image) = 0;
    virtual void visualize(const Features& features) const;
    virtual void setOptions(const Options& options) { options_ = options; }
    const Options& getOptions() const { return options_; }
    
protected:
    Options options_;
};

// Feature tracking module interface
class FeatureTracker : public ModuleBase {
public:
    struct TrackingResult {
        std::vector<cv::Point2f> current_points;
        std::vector<cv::Point2f> prev_points;
        std::vector<uchar> status;
        std::vector<float> errors;
        std::vector<int> track_ids;
        cv::Mat current_image;
        cv::Mat prev_image;
        double tracking_time_ms = 0.0;
        int tracked_count = 0;
        double avg_error = 0.0;
    };
    
    struct Options {
        cv::Size win_size = cv::Size(21, 21);
        int max_level = 3;
        cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        int min_eigen_threshold = 1;
        double max_error = 30.0;
        bool use_initial_flow = false;
    };
    
    virtual ~FeatureTracker() = default;
    
    virtual TrackingResult track(const cv::Mat& prev_img, 
                                const cv::Mat& curr_img,
                                const std::vector<cv::Point2f>& prev_points) = 0;
    virtual void visualize(const TrackingResult& result) const;
    virtual void setOptions(const Options& options) { options_ = options; }
    const Options& getOptions() const { return options_; }
    
protected:
    Options options_;
};

// Pose estimation module interface
class PoseEstimator : public ModuleBase {
public:
    struct PoseResult {
        Eigen::Matrix4d pose;
        std::vector<int> inlier_indices;
        double reprojection_error = 0.0;
        bool is_valid = false;
        int inlier_count = 0;
        double confidence = 0.0;
        cv::Mat image;
        std::vector<cv::Point2f> inlier_points_2d;
        std::vector<cv::Point3f> inlier_points_3d;
        double estimation_time_ms = 0.0;
    };
    
    struct Options {
        int min_points = 10;
        double ransac_threshold = 3.0;
        double confidence = 0.99;
        int max_iterations = 1000;
        bool use_extrinsic_guess = false;
        int solve_pnp_method = cv::SOLVEPNP_ITERATIVE;
    };
    
    virtual ~PoseEstimator() = default;
    
    virtual PoseResult estimate(const std::vector<cv::Point3f>& points_3d,
                               const std::vector<cv::Point2f>& points_2d,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& dist_coeffs,
                               const cv::Mat& image = cv::Mat()) = 0;
    virtual void visualize(const PoseResult& result) const;
    virtual void setOptions(const Options& options) { options_ = options; }
    const Options& getOptions() const { return options_; }
    
protected:
    Options options_;
};

// Depth estimation module interface
class DepthEstimator : public ModuleBase {
public:
    struct DepthResult {
        std::vector<cv::Point3f> points_3d;
        std::vector<double> depths;
        std::vector<double> parallax_angles;
        std::vector<bool> is_valid;
        std::vector<cv::Point2f> points_2d_ref;
        std::vector<cv::Point2f> points_2d_cur;
        cv::Mat reference_image;
        cv::Mat current_image;
        double triangulation_time_ms = 0.0;
        int valid_points_count = 0;
        double avg_depth = 0.0;
        double avg_parallax = 0.0;
    };
    
    struct Options {
        double min_parallax_deg = 1.0;
        double max_parallax_deg = 45.0;
        double min_depth = 0.1;
        double max_depth = 100.0;
        double reprojection_threshold = 2.0;
        bool undistort_points = true;
        int min_triangulation_angle = 1;
    };
    
    virtual ~DepthEstimator() = default;
    
    virtual DepthResult triangulate(const std::vector<cv::Point2f>& points1,
                                  const std::vector<cv::Point2f>& points2,
                                  const Eigen::Matrix4d& pose1,
                                  const Eigen::Matrix4d& pose2,
                                  const cv::Mat& camera_matrix,
                                  const cv::Mat& dist_coeffs,
                                  const cv::Mat& image1 = cv::Mat(),
                                  const cv::Mat& image2 = cv::Mat()) = 0;
    virtual void visualize(const DepthResult& result) const;
    virtual void setOptions(const Options& options) { options_ = options; }
    const Options& getOptions() const { return options_; }
    
protected:
    Options options_;
};

// Bundle adjustment module interface
class BundleAdjuster : public ModuleBase {
public:
    struct BAProblem {
        std::vector<Eigen::Matrix4d> poses;  // Camera poses [world to camera]
        std::vector<Eigen::Vector3d> points;  // 3D points in world frame
        std::vector<std::pair<int, int>> observations;  // (pose_idx, point_idx)
        std::vector<cv::Point2f> measurements;  // 2D measurements
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        std::vector<bool> pose_fixed;  // Which poses to fix
        std::vector<double> wheel_odometry_priors;  // Optional wheel prior weights
    };
    
    struct BAResult {
        std::vector<Eigen::Matrix4d> optimized_poses;
        std::vector<Eigen::Vector3d> optimized_points;
        double initial_error = 0.0;
        double final_error = 0.0;
        int iterations = 0;
        bool success = false;
        std::vector<int> inlier_observations;
        double optimization_time_ms = 0.0;
    };
    
    struct Options {
        int max_iterations = 100;
        bool use_robust_loss = true;
        double huber_parameter = 1.0;
        double function_tolerance = 1e-6;
        double gradient_tolerance = 1e-10;
        double parameter_tolerance = 1e-8;
        bool use_wheel_priors = false;
        double wheel_prior_weight = 1.0;
        bool verbose = false;
    };
    
    virtual ~BundleAdjuster() = default;
    
    virtual BAResult optimize(BAProblem& problem) = 0;
    virtual void visualize(const BAResult& result, const BAProblem& problem) const;
    virtual void setOptions(const Options& options) { options_ = options; }
    const Options& getOptions() const { return options_; }
    
protected:
    Options options_;
};

// Map data structures
struct MapPoint {
    int id = -1;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    cv::Mat descriptor;
    std::vector<int> observed_frame_ids;
    std::vector<cv::Point2f> observations;
    double parallax = 0.0;
    int times_observed = 0;
    bool is_valid = true;
    
    // Quality metrics
    double reprojection_error = 0.0;
    double depth_variance = 0.0;
};

struct Frame {
    int id = -1;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();  // World to camera
    cv::Mat image;
    std::vector<int> map_point_ids;
    bool is_keyframe = false;
    double timestamp = 0.0;
    
    // Features in this frame
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    // For visualization
    std::string image_path;
};

// Result structure for SLAM pipeline
struct SLAMResult {
    std::vector<Frame> frames;
    std::vector<MapPoint> map_points;
    std::vector<Eigen::Matrix4d> camera_poses;  // For backward compatibility
    std::vector<Eigen::Vector3d> points_3d;     // For backward compatibility
    bool success = false;
    std::string error_message;
    double total_processing_time_ms = 0.0;
};

} // namespace slam