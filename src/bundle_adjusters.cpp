#include "slam/bundle_adjusters.hpp"
#include <chrono>

// Use type aliases for cleaner code
using BAResult = slam::BundleAdjuster::BAResult;

namespace slam {

// Ceres Bundle Adjuster Implementation
CeresBundleAdjuster::CeresBundleAdjuster() {
}

BAResult CeresBundleAdjuster::optimize(BAProblem& problem) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BAResult result;
    
    if (problem.poses.empty() || problem.points.empty() || problem.observations.empty()) {
        result.success = false;
        return result;
    }
    
    // Set up Ceres problem
    ceres::Problem ceres_problem;
    
    // Convert poses to parameter blocks (angle-axis + translation)
    std::vector<double> pose_params;
    for (const auto& pose : problem.poses) {
        Eigen::AngleAxisd angle_axis(pose.block<3, 3>(0, 0));
        pose_params.push_back(angle_axis.angle() * angle_axis.axis()(0));
        pose_params.push_back(angle_axis.angle() * angle_axis.axis()(1));
        pose_params.push_back(angle_axis.angle() * angle_axis.axis()(2));
        pose_params.push_back(pose(0, 3));
        pose_params.push_back(pose(1, 3));
        pose_params.push_back(pose(2, 3));
    }
    
    // Convert points to parameter blocks
    std::vector<double> point_params;
    for (const auto& point : problem.points) {
        point_params.push_back(point(0));
        point_params.push_back(point(1));
        point_params.push_back(point(2));
    }
    
    // Get camera parameters
    double fx = problem.camera_matrix.at<double>(0, 0);
    double fy = problem.camera_matrix.at<double>(1, 1);
    double cx = problem.camera_matrix.at<double>(0, 2);
    double cy = problem.camera_matrix.at<double>(1, 2);
    
    // Add residual blocks
    for (size_t i = 0; i < problem.observations.size(); ++i) {
        int pose_idx = problem.observations[i].first;
        int point_idx = problem.observations[i].second;
        const auto& measurement = problem.measurements[i];
        
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(measurement.x, measurement.y, 
                                    fx, fy, cx, cy));
        
        if (options_.use_robust_loss) {
            ceres::LossFunction* loss_function = new ceres::HuberLoss(options_.huber_parameter);
            ceres_problem.AddResidualBlock(cost_function, loss_function,
                                         pose_params.data() + pose_idx * 6,
                                         point_params.data() + point_idx * 3);
        } else {
            ceres_problem.AddResidualBlock(cost_function, nullptr,
                                         pose_params.data() + pose_idx * 6,
                                         point_params.data() + point_idx * 3);
        }
    }
    
    // Set parameter blocks constant if needed
    for (size_t i = 0; i < problem.pose_fixed.size(); ++i) {
        if (problem.pose_fixed[i]) {
            ceres_problem.SetParameterBlockConstant(pose_params.data() + i * 6);
        }
    }
    
    // Set solver options
    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options_.max_iterations;
    solver_options.function_tolerance = options_.function_tolerance;
    solver_options.gradient_tolerance = options_.gradient_tolerance;
    solver_options.parameter_tolerance = options_.parameter_tolerance;
    solver_options.minimizer_progress_to_stdout = options_.verbose;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    
    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &ceres_problem, &summary);
    
    // Extract results
    result.success = summary.IsSolutionUsable();
    result.initial_error = summary.initial_cost;
    result.final_error = summary.final_cost;
    result.iterations = summary.iterations.size();
    
    // Convert back to poses and points
    result.optimized_poses.resize(problem.poses.size());
    for (size_t i = 0; i < problem.poses.size(); ++i) {
        Eigen::Vector3d angle_axis(pose_params[i * 6 + 0], 
                                  pose_params[i * 6 + 1], 
                                  pose_params[i * 6 + 2]);
        Eigen::AngleAxisd rotation(angle_axis.norm(), angle_axis.normalized());
        Eigen::Matrix3d R = rotation.toRotationMatrix();
        
        result.optimized_poses[i] = Eigen::Matrix4d::Identity();
        result.optimized_poses[i].block<3, 3>(0, 0) = R;
        result.optimized_poses[i](0, 3) = pose_params[i * 6 + 3];
        result.optimized_poses[i](1, 3) = pose_params[i * 6 + 4];
        result.optimized_poses[i](2, 3) = pose_params[i * 6 + 5];
    }
    
    result.optimized_points.resize(problem.points.size());
    for (size_t i = 0; i < problem.points.size(); ++i) {
        result.optimized_points[i] = Eigen::Vector3d(
            point_params[i * 3 + 0],
            point_params[i * 3 + 1],
            point_params[i * 3 + 2]
        );
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.optimization_time_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    visualize(result, problem);
    
    return result;
}

template <typename T>
bool CeresBundleAdjuster::ReprojectionError::operator()(const T* const camera_pose, const T* const point, T* residuals) const {
    // Transform point to camera coordinates
    T p[3];
    // Manual angle-axis rotation
    T angle_axis[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
    T norm = sqrt(angle_axis[0]*angle_axis[0] + angle_axis[1]*angle_axis[1] + angle_axis[2]*angle_axis[2]);
    if (norm > T(0)) {
        T c = cos(norm);
        T s = sin(norm);
        T t = T(1) - c;
        T x = angle_axis[0] / norm;
        T y = angle_axis[1] / norm;
        T z = angle_axis[2] / norm;
        
        // Rodrigues' rotation formula
        p[0] = (x*x*t + c) * point[0] + (x*y*t - z*s) * point[1] + (x*z*t + y*s) * point[2];
        p[1] = (y*x*t + z*s) * point[0] + (y*y*t + c) * point[1] + (y*z*t - x*s) * point[2];
        p[2] = (z*x*t - y*s) * point[0] + (z*y*t + x*s) * point[1] + (z*z*t + c) * point[2];
    } else {
        p[0] = point[0];
        p[1] = point[1];
        p[2] = point[2];
    }
    p[0] += camera_pose[3];
    p[1] += camera_pose[4];
    p[2] += camera_pose[5];
    
    // Project to image plane
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    
    // Apply camera intrinsics
    T predicted_x = focal_x_ * xp + principal_x_;
    T predicted_y = focal_y_ * yp + principal_y_;
    
    // Compute residuals
    residuals[0] = predicted_x - T(observed_x_);
    residuals[1] = predicted_y - T(observed_y_);
    
    return true;
}

void CeresBundleAdjuster::setLossFunction(const std::string& /*type*/) {
    // Implementation for setting loss function type
}

void CeresBundleAdjuster::setLinearSolverType(const std::string& /*type*/) {
    // Implementation for setting linear solver type
}

void CeresBundleAdjuster::setUpdateType(const std::string& /*type*/) {
    // Implementation for setting update type
}

void CeresBundleAdjuster::setTrustRegionStrategy(const std::string& /*type*/) {
    // Implementation for setting trust region strategy
}

void CeresBundleAdjuster::setIterationCallback(
    std::function<void(const ceres::IterationSummary&)> callback) {
    iteration_callback_ = callback;
}

// Placeholder implementations for other bundle adjusters
RobustBundleAdjuster::RobustBundleAdjuster() {}
BAResult RobustBundleAdjuster::optimize(BAProblem& /*problem*/) { 
    BAResult result;
    result.success = false;
    return result;
}

IncrementalBundleAdjuster::IncrementalBundleAdjuster() {}
BAResult IncrementalBundleAdjuster::optimize(BAProblem& /*problem*/) { 
    BAResult result;
    result.success = false;
    return result;
}

PoseGraphBundleAdjuster::PoseGraphBundleAdjuster() {}
BAResult PoseGraphBundleAdjuster::optimize(BAProblem& /*problem*/) { 
    BAResult result;
    result.success = false;
    return result;
}

// Factory function implementation
std::unique_ptr<BundleAdjuster> createBundleAdjuster(
    const std::string& type,
    const BundleAdjuster::Options& options) {
    
    std::unique_ptr<BundleAdjuster> ba;
    
    if (type == "Ceres") {
        ba = std::make_unique<CeresBundleAdjuster>();
    } else if (type == "Robust") {
        ba = std::make_unique<RobustBundleAdjuster>();
    } else if (type == "Incremental") {
        ba = std::make_unique<IncrementalBundleAdjuster>();
    } else if (type == "PoseGraph") {
        ba = std::make_unique<PoseGraphBundleAdjuster>();
    } else {
        // Default to Ceres
        ba = std::make_unique<CeresBundleAdjuster>();
    }
    
    ba->setOptions(options);
    return ba;
}

} // namespace slam