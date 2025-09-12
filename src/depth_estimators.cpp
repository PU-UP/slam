#include "slam/depth_estimators.hpp"
#include <chrono>
#include <algorithm>
#include <Eigen/SVD>

// Use type aliases for cleaner code
using DepthResult = slam::DepthEstimator::DepthResult;

namespace slam {

// Standard Triangulator Implementation
StandardTriangulator::StandardTriangulator() 
    : triangulation_method_(0), min_disparity_(1.0), enable_angle_check_(true) {
}

DepthResult StandardTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to normalized image coordinates
    cv::Mat K = camera_matrix.clone();
    cv::Mat K_inv = K.inv();
    
    std::vector<cv::Point2f> norm_points1, norm_points2;
    cv::undistortPoints(points1, norm_points1, K, dist_coeffs);
    cv::undistortPoints(points2, norm_points2, K, dist_coeffs);
    
    // Get projection matrices
    Eigen::Matrix3d K_eigen;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K_eigen(i, j) = K.at<double>(i, j);
        }
    }
    
    Eigen::Matrix4d pose_rel = pose2 * pose1.inverse();
    Eigen::Matrix3d R = pose_rel.block<3, 3>(0, 0);
    Eigen::Vector3d t = pose_rel.block<3, 1>(0, 3);
    
    // Prepare for OpenCV triangulation
    cv::Mat P1(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    
    // P1 = K * [I | 0]
    P1.at<double>(0, 0) = K.at<double>(0, 0); P1.at<double>(0, 1) = 0; P1.at<double>(0, 2) = K.at<double>(0, 2); P1.at<double>(0, 3) = 0;
    P1.at<double>(1, 0) = 0; P1.at<double>(1, 1) = K.at<double>(1, 1); P1.at<double>(1, 2) = K.at<double>(1, 2); P1.at<double>(1, 3) = 0;
    P1.at<double>(2, 0) = 0; P1.at<double>(2, 1) = 0; P1.at<double>(2, 2) = 1; P1.at<double>(2, 3) = 0;
    
    // P2 = K * [R | t]
    cv::Mat R_mat(3, 3, CV_64F);
    cv::Mat t_mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_mat.at<double>(i, j) = R(i, j);
        }
        t_mat.at<double>(i, 0) = t(i);
    }
    
    cv::Mat KR = K * R_mat;
    cv::Mat Kt = K * t_mat;
    P2.at<double>(0, 0) = KR.at<double>(0, 0); P2.at<double>(0, 1) = KR.at<double>(0, 1); P2.at<double>(0, 2) = KR.at<double>(0, 2); P2.at<double>(0, 3) = Kt.at<double>(0, 0);
    P2.at<double>(1, 0) = KR.at<double>(1, 0); P2.at<double>(1, 1) = KR.at<double>(1, 1); P2.at<double>(1, 2) = KR.at<double>(1, 2); P2.at<double>(1, 3) = Kt.at<double>(1, 0);
    P2.at<double>(2, 0) = KR.at<double>(2, 0); P2.at<double>(2, 1) = KR.at<double>(2, 1); P2.at<double>(2, 2) = KR.at<double>(2, 2); P2.at<double>(2, 3) = Kt.at<double>(2, 0);
    
    // Convert to homogeneous coordinates
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1, points2, points4D);
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    int valid_count = 0;
    double total_depth = 0.0;
    double total_parallax = 0.0;
    
    // Convert 4D points to 3D
    for (size_t i = 0; i < points1.size(); ++i) {
        double w = points4D.at<float>(3, i);
        if (std::abs(w) > 1e-7) {
            cv::Point3f pt3d(
                points4D.at<float>(0, i) / w,
                points4D.at<float>(1, i) / w,
                points4D.at<float>(2, i) / w
            );
            
            // Convert to world coordinates
            Eigen::Vector4d point_cam;
            point_cam << pt3d.x, pt3d.y, pt3d.z, 1.0;
            Eigen::Vector4d point_world = pose1 * point_cam;
            
            result.points_3d[i] = cv::Point3f(point_world(0), point_world(1), point_world(2));
            
            // Calculate depth (distance from camera)
            double depth = point_cam.norm();
            result.depths[i] = depth;
            
            // Calculate parallax angle
            Eigen::Vector3d ray1(norm_points1[i].x, norm_points1[i].y, 1.0);
            Eigen::Vector3d ray2(norm_points2[i].x, norm_points2[i].y, 1.0);
            
            ray1 = pose1.block<3, 3>(0, 0) * ray1;
            ray2 = pose2.block<3, 3>(0, 0) * ray2;
            
            double parallax_angle = calculateParallaxAngle(ray1, ray2);
            result.parallax_angles[i] = parallax_angle * 180.0 / M_PI;
            
            // Check validity
            bool valid = true;
            
            // Depth check
            if (depth < options_.min_depth || depth > options_.max_depth) {
                valid = false;
            }
            
            // Parallax check
            if (enable_angle_check_ && 
                (parallax_angle * 180.0 / M_PI < options_.min_parallax_deg ||
                 parallax_angle * 180.0 / M_PI > options_.max_parallax_deg)) {
                valid = false;
            }
            
            // Cheirality check (point in front of both cameras)
            if (point_cam(2) < 0) {
                valid = false;
            }
            
            Eigen::Vector4d point_cam2 = pose2.inverse() * point_world;
            if (point_cam2(2) < 0) {
                valid = false;
            }
            
            result.is_valid[i] = valid;
            
            if (valid) {
                valid_count++;
                total_depth += depth;
                total_parallax += parallax_angle * 180.0 / M_PI;
            }
        }
    }
    
    result.valid_points_count = valid_count;
    result.avg_depth = valid_count > 0 ? total_depth / valid_count : 0.0;
    result.avg_parallax = valid_count > 0 ? total_parallax / valid_count : 0.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

double StandardTriangulator::calculateParallaxAngle(const Eigen::Vector3d& ray1, 
                                                   const Eigen::Vector3d& ray2) {
    double cos_angle = ray1.normalized().dot(ray2.normalized());
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return std::acos(cos_angle);
}

void StandardTriangulator::setMethod(int method) {
    triangulation_method_ = method;
}

void StandardTriangulator::setMinDisparity(double min_disparity) {
    min_disparity_ = min_disparity;
}

void StandardTriangulator::setAngleCheck(bool enable) {
    enable_angle_check_ = enable;
}

// Midpoint Triangulator Implementation
MidpointTriangulator::MidpointTriangulator() 
    : max_iterations_(10), convergence_threshold_(1e-6), use_weights_(true) {
}

DepthResult MidpointTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to normalized coordinates
    cv::Mat K = camera_matrix.clone();
    std::vector<cv::Point2f> norm_points1, norm_points2;
    cv::undistortPoints(points1, norm_points1, K, dist_coeffs);
    cv::undistortPoints(points2, norm_points2, K, dist_coeffs);
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    int valid_count = 0;
    double total_depth = 0.0;
    double total_parallax = 0.0;
    
    Eigen::Matrix3d K_eigen;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K_eigen(i, j) = K.at<double>(i, j);
        }
    }
    
    for (size_t i = 0; i < points1.size(); ++i) {
        // Convert to rays
        Eigen::Vector3d ray1(norm_points1[i].x, norm_points1[i].y, 1.0);
        Eigen::Vector3d ray2(norm_points2[i].x, norm_points2[i].y, 1.0);
        
        // Triangulate using midpoint method
        Eigen::Vector3d point_cam1 = midpointTriangulation(ray1, ray2, pose1, pose2);
        
        // Convert to world coordinates
        Eigen::Vector4d point_cam_homo;
        point_cam_homo << point_cam1, 1.0;
        Eigen::Vector4d point_world = pose1 * point_cam_homo;
        
        result.points_3d[i] = cv::Point3f(point_world(0), point_world(1), point_world(2));
        
        // Calculate depth and parallax
        double depth = point_cam1.norm();
        result.depths[i] = depth;
        
        Eigen::Vector3d ray1_world = pose1.block<3, 3>(0, 0) * ray1;
        Eigen::Vector3d ray2_world = pose2.block<3, 3>(0, 0) * ray2;
        double parallax_angle = std::acos(ray1_world.normalized().dot(ray2_world.normalized()));
        result.parallax_angles[i] = parallax_angle * 180.0 / M_PI;
        
        // Check validity
        bool valid = depth >= options_.min_depth && depth <= options_.max_depth &&
                     parallax_angle * 180.0 / M_PI >= options_.min_parallax_deg &&
                     parallax_angle * 180.0 / M_PI <= options_.max_parallax_deg &&
                     point_cam1(2) > 0;
        
        // Check second camera
        Eigen::Vector4d point_cam2 = pose2.inverse() * point_world;
        if (point_cam2(2) < 0) {
            valid = false;
        }
        
        result.is_valid[i] = valid;
        
        if (valid) {
            valid_count++;
            total_depth += depth;
            total_parallax += parallax_angle * 180.0 / M_PI;
        }
    }
    
    result.valid_points_count = valid_count;
    result.avg_depth = valid_count > 0 ? total_depth / valid_count : 0.0;
    result.avg_parallax = valid_count > 0 ? total_parallax / valid_count : 0.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

Eigen::Vector3d MidpointTriangulator::midpointTriangulation(
    const Eigen::Vector3d& ray1,
    const Eigen::Vector3d& ray2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) {
    
    Eigen::Matrix4d pose_rel = pose2 * pose1.inverse();
    Eigen::Matrix3d R = pose_rel.block<3, 3>(0, 0);
    Eigen::Vector3d t = pose_rel.block<3, 1>(0, 3);
    
    // Transform ray2 to camera 1 coordinate system
    Eigen::Vector3d ray2_cam1 = R.inverse() * ray2;
    Eigen::Vector3d t_cam1 = R.inverse() * t;
    
    // Find closest points between two rays
    Eigen::Vector3d w0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d w1 = t_cam1;
    
    Eigen::Vector3d u = ray1.normalized();
    Eigen::Vector3d v = ray2_cam1.normalized();
    
    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w1 - w0);
    double e = v.dot(w1 - w0);
    
    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-10) {
        // Parallel rays, use average
        return (w0 + w1) / 2.0;
    }
    
    double sc = (b * e - c * d) / denom;
    double tc = (a * e - b * d) / denom;
    
    Eigen::Vector3d pc = w0 + sc * u;
    Eigen::Vector3d qc = w1 + tc * v;
    
    // Return midpoint
    return (pc + qc) / 2.0;
}

void MidpointTriangulator::setMaxIterations(int max_iter) {
    max_iterations_ = max_iter;
}

void MidpointTriangulator::setConvergenceThreshold(double threshold) {
    convergence_threshold_ = threshold;
}

void MidpointTriangulator::setUseWeights(bool use_weights) {
    use_weights_ = use_weights;
}

// DLT Triangulator Implementation
DLTTriangulator::DLTTriangulator() 
    : normalize_points_(true), use_ransac_(false), ransac_threshold_(3.0) {
}

DepthResult DLTTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& /*dist_coeffs*/,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert camera matrix to Eigen
    Eigen::Matrix3d K;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K(i, j) = camera_matrix.at<double>(i, j);
        }
    }
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    int valid_count = 0;
    double total_depth = 0.0;
    double total_parallax = 0.0;
    
    for (size_t i = 0; i < points1.size(); ++i) {
        // Triangulate using DLT
        Eigen::Vector3d point_cam = dltTriangulation(points1[i], points2[i], K, K, pose1, pose2);
        
        // Convert to world coordinates
        Eigen::Vector4d point_cam_homo;
        point_cam_homo << point_cam, 1.0;
        Eigen::Vector4d point_world = pose1 * point_cam_homo;
        
        result.points_3d[i] = cv::Point3f(point_world(0), point_world(1), point_world(2));
        
        // Calculate depth
        double depth = point_cam.norm();
        result.depths[i] = depth;
        
        // Calculate parallax angle
        Eigen::Vector3d ray1, ray2;
        ray1 << points1[i].x, points1[i].y, 1.0;
        ray2 << points2[i].x, points2[i].y, 1.0;
        
        ray1 = K.inverse() * ray1;
        ray2 = K.inverse() * ray2;
        
        ray1 = pose1.block<3, 3>(0, 0) * ray1;
        ray2 = pose2.block<3, 3>(0, 0) * ray2;
        
        double parallax_angle = std::acos(ray1.normalized().dot(ray2.normalized()));
        result.parallax_angles[i] = parallax_angle * 180.0 / M_PI;
        
        // Check validity
        bool valid = depth >= options_.min_depth && depth <= options_.max_depth &&
                     parallax_angle * 180.0 / M_PI >= options_.min_parallax_deg &&
                     parallax_angle * 180.0 / M_PI <= options_.max_parallax_deg &&
                     point_cam(2) > 0;
        
        // Check second camera
        Eigen::Vector4d point_cam2 = pose2.inverse() * point_world;
        if (point_cam2(2) < 0) {
            valid = false;
        }
        
        result.is_valid[i] = valid;
        
        if (valid) {
            valid_count++;
            total_depth += depth;
            total_parallax += parallax_angle * 180.0 / M_PI;
        }
    }
    
    result.valid_points_count = valid_count;
    result.avg_depth = valid_count > 0 ? total_depth / valid_count : 0.0;
    result.avg_parallax = valid_count > 0 ? total_parallax / valid_count : 0.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

cv::Mat DLTTriangulator::normalizePoints(const std::vector<cv::Point2f>& points,
                                        std::vector<cv::Point2f>& normalized_points,
                                        cv::Mat& T) {
    // Calculate centroid
    cv::Scalar centroid = cv::mean(points);
    double cx = centroid[0];
    double cy = centroid[1];
    
    // Calculate mean distance
    double dist = 0.0;
    for (const auto& pt : points) {
        dist += std::sqrt((pt.x - cx) * (pt.x - cx) + (pt.y - cy) * (pt.y - cy));
    }
    dist /= points.size();
    
    // Scale factor
    double scale = std::sqrt(2.0) / dist;
    
    // Create transformation matrix
    T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 0) = scale;
    T.at<double>(1, 1) = scale;
    T.at<double>(0, 2) = -scale * cx;
    T.at<double>(1, 2) = -scale * cy;
    
    // Normalize points
    normalized_points.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        normalized_points[i].x = scale * (points[i].x - cx);
        normalized_points[i].y = scale * (points[i].y - cy);
    }
    
    return T;
}

Eigen::Vector3d DLTTriangulator::dltTriangulation(
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const Eigen::Matrix3d& K1,
    const Eigen::Matrix3d& K2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) {
    
    // Convert to homogeneous coordinates
    Eigen::Vector3d x1, x2;
    x1 << pt1.x, pt1.y, 1.0;
    x2 << pt2.x, pt2.y, 1.0;
    
    // Get projection matrices
    Eigen::Matrix3d R1 = pose1.block<3, 3>(0, 0);
    Eigen::Vector3d t1 = pose1.block<3, 1>(0, 3);
    Eigen::Matrix3d R2 = pose2.block<3, 3>(0, 0);
    Eigen::Vector3d t2 = pose2.block<3, 1>(0, 3);
    
    Eigen::Matrix<double, 3, 4> P1 = K1 * Eigen::Matrix<double, 3, 4>::Identity();
    P1.block<3, 3>(0, 0) = K1 * R1;
    P1.block<3, 1>(0, 3) = K1 * t1;
    
    Eigen::Matrix<double, 3, 4> P2 = Eigen::Matrix<double, 3, 4>::Identity();
    P2.block<3, 3>(0, 0) = K2 * R2;
    P2.block<3, 1>(0, 3) = K2 * t2;
    
    // Build DLT matrix
    Eigen::Matrix4d A;
    A.row(0) = x1(0) * P1.row(2) - P1.row(0);
    A.row(1) = x1(1) * P1.row(2) - P1.row(1);
    A.row(2) = x2(0) * P2.row(2) - P2.row(0);
    A.row(3) = x2(1) * P2.row(2) - P2.row(1);
    
    // Solve using SVD
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    
    return X.head<3>() / X(3);
}

void DLTTriangulator::setNormalizePoints(bool normalize) {
    normalize_points_ = normalize;
}

void DLTTriangulator::setUseRansac(bool use_ransac) {
    use_ransac_ = use_ransac;
}

void DLTTriangulator::setRansacThreshold(double threshold) {
    ransac_threshold_ = threshold;
}

// Optimal Triangulator Implementation
OptimalTriangulator::OptimalTriangulator() 
    : use_ceres_(true), max_iterations_(50), huber_threshold_(1.0), convergence_threshold_(1e-6) {
}

DepthResult OptimalTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& /*dist_coeffs*/,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert camera matrix to Eigen
    Eigen::Matrix3d K;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K(i, j) = camera_matrix.at<double>(i, j);
        }
    }
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    int valid_count = 0;
    double total_depth = 0.0;
    double total_parallax = 0.0;
    
    for (size_t i = 0; i < points1.size(); ++i) {
        // First get initial estimate using DLT
        DLTTriangulator dlt;
        Eigen::Vector3d point_cam = dlt.dltTriangulation(points1[i], points2[i], K, K, pose1, pose2);
        
        // Refine using optimal triangulation
        if (use_ceres_) {
            point_cam = optimalTriangulationCeres(points1[i], points2[i], K, pose1, pose2);
        } else {
            point_cam = optimalTriangulationGaussNewton(points1[i], points2[i], K, pose1, pose2);
        }
        
        // Convert to world coordinates
        Eigen::Vector4d point_cam_homo;
        point_cam_homo << point_cam, 1.0;
        Eigen::Vector4d point_world = pose1 * point_cam_homo;
        
        result.points_3d[i] = cv::Point3f(point_world(0), point_world(1), point_world(2));
        
        // Calculate depth
        double depth = point_cam.norm();
        result.depths[i] = depth;
        
        // Calculate parallax angle
        Eigen::Vector3d ray1, ray2;
        ray1 << points1[i].x, points1[i].y, 1.0;
        ray2 << points2[i].x, points2[i].y, 1.0;
        
        ray1 = K.inverse() * ray1;
        ray2 = K.inverse() * ray2;
        
        ray1 = pose1.block<3, 3>(0, 0) * ray1;
        ray2 = pose2.block<3, 3>(0, 0) * ray2;
        
        double parallax_angle = std::acos(ray1.normalized().dot(ray2.normalized()));
        result.parallax_angles[i] = parallax_angle * 180.0 / M_PI;
        
        // Check validity
        bool valid = depth >= options_.min_depth && depth <= options_.max_depth &&
                     parallax_angle * 180.0 / M_PI >= options_.min_parallax_deg &&
                     parallax_angle * 180.0 / M_PI <= options_.max_parallax_deg &&
                     point_cam(2) > 0;
        
        // Check second camera
        Eigen::Vector4d point_cam2 = pose2.inverse() * point_world;
        if (point_cam2(2) < 0) {
            valid = false;
        }
        
        result.is_valid[i] = valid;
        
        if (valid) {
            valid_count++;
            total_depth += depth;
            total_parallax += parallax_angle * 180.0 / M_PI;
        }
    }
    
    result.valid_points_count = valid_count;
    result.avg_depth = valid_count > 0 ? total_depth / valid_count : 0.0;
    result.avg_parallax = valid_count > 0 ? total_parallax / valid_count : 0.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

template <typename T>
bool OptimalTriangulator::TriangulationCostFunctor::operator()(const T* const point, T* residual) const {
    // Transform point to camera coordinates
    Eigen::Matrix<T, 4, 1> point_homo;
    point_homo << point[0], point[1], point[2], T(1.0);
    
    // Project to first camera
    Eigen::Matrix<T, 4, 1> point_cam1 = pose1_.cast<T>() * point_homo;
    Eigen::Matrix<T, 3, 1> proj1 = K1_.cast<T>() * point_cam1.hnormalized();
    
    // Project to second camera
    Eigen::Matrix<T, 4, 1> point_cam2 = pose2_.cast<T>() * point_homo;
    Eigen::Matrix<T, 3, 1> proj2 = K2_.cast<T>() * point_cam2.hnormalized();
    
    // Compute reprojection errors
    residual[0] = proj1(0) - T(pt1_.x);
    residual[1] = proj1(1) - T(pt1_.y);
    residual[2] = proj2(0) - T(pt2_.x);
    residual[3] = proj2(1) - T(pt2_.y);
    
    return true;
}

Eigen::Vector3d OptimalTriangulator::optimalTriangulationCeres(
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) {
    
    // Get initial estimate
    DLTTriangulator dlt;
    Eigen::Vector3d initial_point = dlt.dltTriangulation(pt1, pt2, K, K, pose1, pose2);
    
    // Set up Ceres problem
    ceres::Problem problem;
    
    double point[3] = {initial_point(0), initial_point(1), initial_point(2)};
    
    TriangulationCostFunctor* cost_functor = 
        new TriangulationCostFunctor(pt1, pt2, K, K, pose1, pose2);
    
    ceres::CostFunction* cost_function = 
        new ceres::AutoDiffCostFunction<TriangulationCostFunctor, 4, 3>(cost_functor);
    
    problem.AddResidualBlock(cost_function, nullptr, point);
    
    // Set solver options
    ceres::Solver::Options options;
    options.max_num_iterations = max_iterations_;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    
    if (huber_threshold_ > 0) {
        ceres::LossFunction* loss_function = new ceres::HuberLoss(huber_threshold_);
        problem.AddResidualBlock(cost_function, loss_function, point);
    }
    
    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    return Eigen::Vector3d(point[0], point[1], point[2]);
}

Eigen::Vector3d OptimalTriangulator::optimalTriangulationGaussNewton(
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) {
    
    // Get initial estimate
    DLTTriangulator dlt;
    Eigen::Vector3d point = dlt.dltTriangulation(pt1, pt2, K, K, pose1, pose2);
    
    // Gauss-Newton optimization
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Compute residuals and Jacobian
        Eigen::Vector4d residual;
        Eigen::Matrix<double, 4, 3> J;
        
        // Transform point to camera coordinates
        Eigen::Vector4d point_homo;
        point_homo << point, 1.0;
        
        // Project to first camera
        Eigen::Vector4d point_cam1 = pose1 * point_homo;
        Eigen::Vector3d proj1 = K * point_cam1.hnormalized();
        
        // Project to second camera
        Eigen::Vector4d point_cam2 = pose2 * point_homo;
        Eigen::Vector3d proj2 = K * point_cam2.hnormalized();
        
        // Residuals
        residual(0) = proj1(0) - pt1.x;
        residual(1) = proj1(1) - pt1.y;
        residual(2) = proj2(0) - pt2.x;
        residual(3) = proj2(1) - pt2.y;
        
        // Check convergence
        if (residual.norm() < convergence_threshold_) {
            break;
        }
        
        // Compute Jacobian (numerical differentiation)
        double eps = 1e-6;
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d point_plus = point;
            point_plus(i) += eps;
            
            Eigen::Vector4d point_plus_homo;
            point_plus_homo << point_plus, 1.0;
            
            Eigen::Vector4d point_cam1_plus = pose1 * point_plus_homo;
            Eigen::Vector3d proj1_plus = K * point_cam1_plus.hnormalized();
            Eigen::Vector4d point_cam2_plus = pose2 * point_plus_homo;
            Eigen::Vector3d proj2_plus = K * point_cam2_plus.hnormalized();
            
            Eigen::Vector4d residual_plus;
            residual_plus(0) = proj1_plus(0) - pt1.x;
            residual_plus(1) = proj1_plus(1) - pt1.y;
            residual_plus(2) = proj2_plus(0) - pt2.x;
            residual_plus(3) = proj2_plus(1) - pt2.y;
            
            J.col(i) = (residual_plus - residual) / eps;
        }
        
        // Update
        Eigen::Vector3d delta = (J.transpose() * J).inverse() * J.transpose() * residual;
        point -= delta;
    }
    
    return point;
}

void OptimalTriangulator::setUseCeres(bool use_ceres) {
    use_ceres_ = use_ceres;
}

void OptimalTriangulator::setMaxIterations(int max_iter) {
    max_iterations_ = max_iter;
}

void OptimalTriangulator::setHuberThreshold(double threshold) {
    huber_threshold_ = threshold;
}

// Multi-frame Triangulator Implementation
MultiFrameTriangulator::MultiFrameTriangulator() 
    : min_observations_(2), outlier_threshold_(3.0) {
}

DepthResult MultiFrameTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Add the two main observations
    observations_.clear();
    for (size_t i = 0; i < points1.size(); ++i) {
        observations_.push_back({points1[i], pose1});
        observations_.push_back({points2[i], pose2});
    }
    
    // Use optimal triangulation for each pair
    OptimalTriangulator optimal;
    optimal.setOptions(options_);
    
    // Initialize with the best pair (largest parallax)
    double max_parallax = 0.0;
    int best_idx1 = 0, best_idx2 = 1;
    
    for (size_t i = 0; i < observations_.size(); ++i) {
        for (size_t j = i + 1; j < observations_.size(); ++j) {
            Eigen::Matrix4d pose_rel = observations_[j].pose * observations_[i].pose.inverse();
            double translation = pose_rel.block<3, 1>(0, 3).norm();
            if (translation > max_parallax) {
                max_parallax = translation;
                best_idx1 = i;
                best_idx2 = j;
            }
        }
    }
    
    // Get initial triangulation from best pair
    auto init_result = optimal.triangulate(
        {observations_[best_idx1].point}, {observations_[best_idx2].point},
        observations_[best_idx1].pose, observations_[best_idx2].pose,
        camera_matrix, dist_coeffs, image1, image2);
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    if (!init_result.points_3d.empty() && init_result.is_valid[0]) {
        result.points_3d[0] = init_result.points_3d[0];
        result.depths[0] = init_result.depths[0];
        result.parallax_angles[0] = init_result.parallax_angles[0];
        result.is_valid[0] = true;
    }
    
    // Note: This is a simplified implementation
    // A full multi-frame implementation would bundle-adjust all observations
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

void MultiFrameTriangulator::addObservation(const cv::Point2f& point, const Eigen::Matrix4d& pose) {
    observations_.push_back({point, pose});
}

void MultiFrameTriangulator::clearObservations() {
    observations_.clear();
}

void MultiFrameTriangulator::setMinObservations(int min_obs) {
    min_observations_ = min_obs;
}

void MultiFrameTriangulator::setOutlierThreshold(double threshold) {
    outlier_threshold_ = threshold;
}

std::vector<bool> MultiFrameTriangulator::outlierRejection(
    const std::vector<Eigen::Vector3d>& /*points*/,
    const std::vector<Observation>& obs) {
    
    // Simple outlier rejection based on reprojection error
    std::vector<bool> inliers(obs.size(), true);
    
    // This would be implemented with proper outlier rejection
    // For now, return all inliers
    
    return inliers;
}

// Patch Stereo Triangulator Implementation
PatchStereoTriangulator::PatchStereoTriangulator() 
    : patch_size_(15), min_disparity_(0), max_disparity_(64),
      subpixel_refinement_(true), uniqueness_ratio_(0.15) {
}

DepthResult PatchStereoTriangulator::triangulate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Mat& image1,
    const cv::Mat& image2) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DepthResult result;
    result.reference_image = image1.clone();
    result.current_image = image2.clone();
    result.points_2d_ref = points1;
    result.points_2d_cur = points2;
    
    if (points1.empty() || points2.empty() || points1.size() != points2.size() ||
        image1.empty() || image2.empty()) {
        result.triangulation_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        return result;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray1, gray2;
    if (image1.channels() == 3) {
        cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = image1;
    }
    
    if (image2.channels() == 3) {
        cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray2 = image2;
    }
    
    // Initialize result vectors
    result.points_3d.resize(points1.size());
    result.depths.resize(points1.size());
    result.parallax_angles.resize(points1.size());
    result.is_valid.resize(points1.size(), false);
    
    int valid_count = 0;
    double total_depth = 0.0;
    double total_parallax = 0.0;
    
    // Use standard triangulation as baseline, then refine with patch matching
    StandardTriangulator standard;
    standard.setOptions(options_);
    
    auto baseline_result = standard.triangulate(points1, points2, pose1, pose2,
                                               camera_matrix, dist_coeffs, image1, image2);
    
    for (size_t i = 0; i < points1.size(); ++i) {
        if (!baseline_result.is_valid[i]) {
            continue;
        }
        
        // Refine using patch matching
        cv::Point2f pt1 = points1[i];
        cv::Point2f pt2_estimate = points2[i];
        
        // Extract patch from first image
        int half_patch = patch_size_ / 2;
        cv::Rect patch_roi(pt1.x - half_patch, pt1.y - half_patch, 
                          patch_size_, patch_size_);
        
        if (patch_roi.x < 0 || patch_roi.y < 0 ||
            patch_roi.x + patch_size_ > gray1.cols ||
            patch_roi.y + patch_size_ > gray1.rows) {
            continue;
        }
        
        cv::Mat patch1 = gray1(patch_roi).clone();
        
        // Search along epipolar line in second image
        double best_score = -1.0;
        cv::Point2f best_match = pt2_estimate;
        
        for (int d = -max_disparity_; d <= max_disparity_; ++d) {
            cv::Point2f candidate = pt2_estimate + cv::Point2f(d, 0);
            
            // Check bounds
            cv::Rect candidate_roi(candidate.x - half_patch, candidate.y - half_patch,
                                 patch_size_, patch_size_);
            
            if (candidate_roi.x < 0 || candidate_roi.y < 0 ||
                candidate_roi.x + patch_size_ > gray2.cols ||
                candidate_roi.y + patch_size_ > gray2.rows) {
                continue;
            }
            
            cv::Mat patch2 = gray2(candidate_roi);
            
            // Compute patch similarity
            double score = computePatchMatch(patch1, patch2);
            
            if (score > best_score) {
                best_score = score;
                best_match = candidate;
            }
        }
        
        // Subpixel refinement
        if (subpixel_refinement_) {
            best_match = subpixelRefinement(gray2, best_match);
        }
        
        // Triangulate with refined match
        std::vector<cv::Point2f> refined_pt1 = {pt1};
        std::vector<cv::Point2f> refined_pt2 = {best_match};
        
        auto refined_result = standard.triangulate(refined_pt1, refined_pt2, pose1, pose2,
                                                  camera_matrix, dist_coeffs, image1, image2);
        
        if (!refined_result.points_3d.empty() && refined_result.is_valid[0]) {
            result.points_3d[i] = refined_result.points_3d[0];
            result.depths[i] = refined_result.depths[0];
            result.parallax_angles[i] = refined_result.parallax_angles[0];
            result.is_valid[i] = true;
            
            valid_count++;
            total_depth += result.depths[i];
            total_parallax += result.parallax_angles[i];
        }
    }
    
    result.valid_points_count = valid_count;
    result.avg_depth = valid_count > 0 ? total_depth / valid_count : 0.0;
    result.avg_parallax = valid_count > 0 ? total_parallax / valid_count : 0.0;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.triangulation_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result);
    
    return result;
}

double PatchStereoTriangulator::computePatchMatch(const cv::Mat& patch1, const cv::Mat& patch2) {
    // Zero-mean normalized cross-correlation
    cv::Mat p1, p2;
    patch1.convertTo(p1, CV_32F);
    patch2.convertTo(p2, CV_32F);
    
    // Remove mean
    cv::Scalar mean1 = cv::mean(p1);
    cv::Scalar mean2 = cv::mean(p2);
    p1 -= mean1;
    p2 -= mean2;
    
    // Compute NCC
    double numerator = cv::sum(p1.mul(p2))[0];
    double denominator = std::sqrt(cv::sum(p1.mul(p1))[0] * cv::sum(p2.mul(p2))[0]);
    
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

cv::Point2f PatchStereoTriangulator::subpixelRefinement(const cv::Mat& image, const cv::Point2f& point) {
    // Simple quadratic interpolation for subpixel refinement
    float x = point.x;
    float y = point.y;
    
    if (x < 1 || x >= image.cols - 1 || y < 1 || y >= image.rows - 1) {
        return point;
    }
    
    // Get 3x3 neighborhood
    cv::Mat patch;
    cv::getRectSubPix(image, cv::Size(3, 3), point, patch);
    
    if (patch.depth() != CV_32F) {
        patch.convertTo(patch, CV_32F);
    }
    
    // Fit quadratic function
    float fx = (patch.at<float>(0, 1) - patch.at<float>(2, 1)) / 2.0f;
    float fy = (patch.at<float>(1, 0) - patch.at<float>(1, 2)) / 2.0f;
    float fxx = patch.at<float>(0, 1) - 2 * patch.at<float>(1, 1) + patch.at<float>(2, 1);
    float fyy = patch.at<float>(1, 0) - 2 * patch.at<float>(1, 1) + patch.at<float>(1, 2);
    
    if (std::abs(fxx) < 1e-6 || std::abs(fyy) < 1e-6) {
        return point;
    }
    
    float dx = -fx / fxx;
    float dy = -fy / fyy;
    
    // Limit refinement to [-0.5, 0.5]
    dx = std::max(-0.5f, std::min(0.5f, dx));
    dy = std::max(-0.5f, std::min(0.5f, dy));
    
    return cv::Point2f(x + dx, y + dy);
}

void PatchStereoTriangulator::setPatchSize(int size) {
    patch_size_ = size;
}

void PatchStereoTriangulator::setDisparityRange(int min_disp, int max_disp) {
    min_disparity_ = min_disp;
    max_disparity_ = max_disp;
}

void PatchStereoTriangulator::setSubpixelRefinement(bool enable) {
    subpixel_refinement_ = enable;
}

void PatchStereoTriangulator::setUniquenessRatio(double ratio) {
    uniqueness_ratio_ = ratio;
}

// Factory function implementation
std::unique_ptr<DepthEstimator> createDepthEstimator(
    const std::string& type,
    const DepthEstimator::Options& options) {
    
    std::unique_ptr<DepthEstimator> estimator;
    
    if (type == "Standard") {
        estimator = std::make_unique<StandardTriangulator>();
    } else if (type == "Midpoint") {
        estimator = std::make_unique<MidpointTriangulator>();
    } else if (type == "DLT") {
        estimator = std::make_unique<DLTTriangulator>();
    } else if (type == "Optimal") {
        estimator = std::make_unique<OptimalTriangulator>();
    } else if (type == "MultiFrame") {
        estimator = std::make_unique<MultiFrameTriangulator>();
    } else if (type == "PatchStereo") {
        estimator = std::make_unique<PatchStereoTriangulator>();
    } else {
        // Default to standard triangulation
        estimator = std::make_unique<StandardTriangulator>();
    }
    
    estimator->setOptions(options);
    return estimator;
}

} // namespace slam