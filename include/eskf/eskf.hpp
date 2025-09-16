#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

/**
 * Error-State Kalman Filter for differential-drive robots.
 * 
 * This ESKF implementation is designed for ground vehicles with differential-drive
 * kinematics, using the wheel frame as the vehicle's body frame.
 * 
 * Coordinate Frames:
 *  - World (w): Inertial frame with gravity [0,0,-9.81] (ENU convention)
 *  - Wheel: Vehicle frame located at differential drive center
 *  - IMU (i): Sensor frame at arbitrary location on the robot
 * 
 *  frameA_T_frameB: from frameB to frameA !!!!!
 * 
 * State Vector (15-dimensional):
 *  - Position: p_w (3D, world frame)
 *  - Velocity: v_w (3D, world frame) 
 *  - Orientation: q_wheel (quaternion, world->wheel transformation)
 *  - Gyroscope bias: bg_i (3D, IMU frame)
 *  - Accelerometer bias: ba_i (3D, IMU frame)
 * 
 * Key Features:
 * - Automatic initialization during static periods
 * - Zero-velocity update (ZUPT) for drift correction
 * - Non-holonomic constraints for ground vehicles
 * - Wheel speed measurements for forward velocity updates
 * - IMU bias estimation and compensation
 * - Lever-arm effect compensation (optional)
 */

namespace eskf {

// ---------------- Mathematical Utilities ----------------
/**
 * Create skew-symmetric matrix from 3D vector.
 * @param vec 3D vector
 * @return 3x3 skew-symmetric matrix
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d S;
    S <<     0, -vec.z(),  vec.y(),
          vec.z(),     0, -vec.x(),
         -vec.y(),  vec.x(),     0;
    return S;
}

/**
 * Right-multiplication quaternion update.
 * @param q Current quaternion
 * @param dtheta Small rotation vector
 * @return Updated quaternion
 */
inline Eigen::Quaterniond quaternionRightUpdate(const Eigen::Quaterniond& q, const Eigen::Vector3d& dtheta) {
    Eigen::Quaterniond dq(1, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
    return (q * dq).normalized();
}

// ---------------- Configuration and State Structures ----------------
/**
 * ESKF configuration parameters.
 */
struct FilterConfig {
    // Gravity vector in world frame (ENU convention)
    Eigen::Vector3d gravity_world = {0, 0, -9.81};

    // IMU noise parameters (square root of power spectral density)
    double gyroscope_noise_density = 1.5e-3;   // rad/s/√Hz
    double accelerometer_noise_density = 2.5e-1; // m/s²/√Hz
    double gyroscope_random_walk = 1.0e-5;      // rad/s²/√Hz
    double accelerometer_random_walk = 1.0e-3;   // m/s³/√Hz

    // Wheel speed measurement parameters
    double wheel_speed_noise_std = 0.0001;      // m/s
    double wheel_speed_scale_factor = 1.0;      // dimensionless

    // Extrinsic calibration: wheel frame <- IMU frame
    Eigen::Isometry3d transform_wheel_T_imu = Eigen::Isometry3d::Identity();

    // Zero-velocity update (ZUPT) detection parameters
    double zupt_velocity_threshold = 0.03;      // m/s
    double zupt_gyroscope_threshold = 0.02;     // rad/s
    double zupt_acceleration_threshold = 0.06;  // m/s²
    double zupt_minimum_duration = 1.5;         // seconds
    size_t zupt_minimum_samples = 120;          // samples
    double wheel_data_timeout = 0.5;             // seconds
    double zupt_velocity_noise_std = 0.02;      // m/s
    double zupt_bias_nudging_factor = 0.03;     // dimensionless

    // Non-holonomic constraint parameters
    double nhc_velocity_threshold = 0.1;         // m/s
    double nhc_lateral_noise_std = 0.05;        // m/s

    // Lever-arm compensation
    bool enable_lever_arm_compensation = false;
};

/**
 * Nominal state vector (PVQ + biases).
 * 
 * Standard navigation state format: Position, Velocity, Orientation (Quaternion) + IMU biases
 */
struct NominalState {
    double timestamp = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();          // world frame (P)
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();          // world frame (V)
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity(); // world->wheel (Q)
    Eigen::Vector3d gyroscope_bias = Eigen::Vector3d::Zero();     // IMU frame (bg)
    Eigen::Vector3d accelerometer_bias = Eigen::Vector3d::Zero(); // IMU frame (ba)
};

/**
 * Error state vector and covariance.
 */
struct ErrorState {
    static constexpr int STATE_DIMENSION = 15;
    
    // Error state vector: [dp, dv, dtheta, dbg, dba]
    Eigen::Matrix<double, STATE_DIMENSION, 1> vector = 
        Eigen::Matrix<double, STATE_DIMENSION, 1>::Zero();
    
    // Error state covariance matrix
    Eigen::Matrix<double, STATE_DIMENSION, STATE_DIMENSION> covariance = 
        Eigen::Matrix<double, STATE_DIMENSION, STATE_DIMENSION>::Identity() * 1e-3;
};

/**
 * Wheel speed measurement.
 */
struct WheelSpeedMeasurement {
    double timestamp;
    double speed_raw;  // raw wheel speed
    double speed_scaled;  // speed after scaling
};

/**
 * RTK GPS measurement placeholder.
 */
struct GpsMeasurement {
    double timestamp = 0;
    Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Identity() * 0.25;
    bool has_position_fix = true;
};

// ---------------- Error-State Kalman Filter Class ----------------
class ErrorStateKalmanFilter {
public:
    explicit ErrorStateKalmanFilter(const FilterConfig& config);
    
    // System initialization
    void initializeState(const NominalState& initial_state, 
                        const ErrorState& initial_error_state = ErrorState());
    bool isInitialized() const;
    bool isWaitingForInitialization() const;
    
    // State accessors
    const NominalState& getNominalState() const;
    const ErrorState& getErrorState() const;
    
    // Configuration modifiers
    void setWheelSpeedScaleFactor(double scale_factor);
    void setWheelToImuTransform(const Eigen::Isometry3d& transform);
    
    // Prediction step with IMU measurements
    void predictIMU(double timestamp, 
                   const Eigen::Vector3d& gyroscope_raw,
                   const Eigen::Vector3d& accelerometer_raw);
    
    // Update steps with sensor measurements
    double updateWheelSpeed(double timestamp, double wheel_speed_raw);
    double updateWheelSpeed(double wheel_speed_raw);  // uses current timestamp
    void updateGPS(const GpsMeasurement& gps_measurement);
    
private:
    // Covariance propagation
    void propagateCovariance_(double time_step,
                              const Eigen::Vector3d& acceleration_wheel,
                              const Eigen::Matrix3d& rotation_world_T_wheel);
    
    // Error state injection and reset
    void injectAndResetErrorState_();
    
    // Static initialization procedure
    void performStaticInitialization_();
    
    // Zero-velocity update
    void applyZeroVelocityUpdate_();
    
    // Bias nudging during static periods
    void nudgeBiasesDuringStaticPeriod_();
    
    // Non-holonomic constraint enforcement
    void applyNonHolonomicConstraints_();
    
    // Static detection and window management
    void updateImuStaticDetection_(double timestamp,
                                   const Eigen::Vector3d& gyroscope_raw,
                                   const Eigen::Vector3d& accelerometer_raw);
    void updateWheelStaticDetection_(double timestamp, bool is_wheel_static);
    void tryTriggerInitializationOrZupt_(double current_timestamp);
    void clearInitializationBuffer_();
    void clearStaticWindow_();
    
    // Member variables
    FilterConfig config_;
    NominalState nominal_state_;
    ErrorState error_state_;
    bool is_initialized_ = false;
    bool is_waiting_for_init_ = true;
    
    // Extrinsic transformation cache
    Eigen::Matrix3d rotation_wheel_T_imu_;
    Eigen::Matrix3d rotation_imu_T_wheel_;
    Eigen::Vector3d translation_imu_in_wheel_;
    
    // Angular velocity cache (wheel frame)
    Eigen::Vector3d angular_velocity_wheel_last_;
    Eigen::Vector3d angular_velocity_wheel_previous_;
    bool has_previous_angular_velocity_ = false;
    
    // Initialization buffers
    double initialization_start_time_ = -1.0;
    std::vector<Eigen::Vector3d> initialization_acceleration_buffer_;
    std::vector<Eigen::Vector3d> initialization_gyroscope_buffer_;
    
    // Static window state
    bool is_imu_static_ = false;
    double imu_last_timestamp_ = -1.0;
    bool is_wheel_static_ = false;
    double wheel_last_timestamp_ = -1.0;
    double wheel_last_scaled_speed_ = 0.0;
    double static_window_start_time_ = -1.0;
    size_t static_window_sample_count_ = 0;
    std::vector<Eigen::Vector3d> zupt_acceleration_buffer_;
    std::vector<Eigen::Vector3d> zupt_gyroscope_buffer_;
};

  // ---------------- Implementation ----------------

inline ErrorStateKalmanFilter::ErrorStateKalmanFilter(const FilterConfig& config) : config_(config) {
    rotation_wheel_T_imu_ = config_.transform_wheel_T_imu.linear();
    rotation_imu_T_wheel_ = rotation_wheel_T_imu_.transpose();
    translation_imu_in_wheel_ = config_.transform_wheel_T_imu.translation();
}

inline void ErrorStateKalmanFilter::initializeState(const NominalState& initial_state, 
                                                   const ErrorState& initial_error_state) {
    nominal_state_ = initial_state;
    error_state_ = initial_error_state;
    is_initialized_ = true;
    is_waiting_for_init_ = false;
    clearInitializationBuffer_();
    clearStaticWindow_();
    has_previous_angular_velocity_ = false;
}

inline bool ErrorStateKalmanFilter::isInitialized() const {
    return is_initialized_;
}

inline bool ErrorStateKalmanFilter::isWaitingForInitialization() const {
    return is_waiting_for_init_;
}

inline const NominalState& ErrorStateKalmanFilter::getNominalState() const {
    return nominal_state_;
}

inline const ErrorState& ErrorStateKalmanFilter::getErrorState() const {
    return error_state_;
}

inline void ErrorStateKalmanFilter::setWheelSpeedScaleFactor(double scale_factor) {
    config_.wheel_speed_scale_factor = scale_factor;
}

inline void ErrorStateKalmanFilter::setWheelToImuTransform(const Eigen::Isometry3d& transform) {
    config_.transform_wheel_T_imu = transform;
    rotation_wheel_T_imu_ = transform.linear();
    rotation_imu_T_wheel_ = rotation_wheel_T_imu_.transpose();
    translation_imu_in_wheel_ = transform.translation();
}

inline void ErrorStateKalmanFilter::predictIMU(double timestamp, 
                                              const Eigen::Vector3d& gyroscope_raw,
                                              const Eigen::Vector3d& accelerometer_raw) {
    // Update IMU static detection buffers
    updateImuStaticDetection_(timestamp, gyroscope_raw, accelerometer_raw);

    if (!is_initialized_) {
        tryTriggerInitializationOrZupt_(timestamp);
        nominal_state_.timestamp = timestamp;
        return;
    }

    const double dt = std::max(1e-6, timestamp - nominal_state_.timestamp);
    nominal_state_.timestamp = timestamp;

    // Remove sensor biases
    const Eigen::Vector3d gyroscope_corrected = gyroscope_raw - nominal_state_.gyroscope_bias;
    const Eigen::Vector3d accelerometer_corrected = accelerometer_raw - nominal_state_.accelerometer_bias;

    // Transform measurements to wheel frame
    const Eigen::Vector3d angular_velocity_wheel = rotation_wheel_T_imu_ * gyroscope_corrected;
    Eigen::Vector3d acceleration_wheel = rotation_wheel_T_imu_ * accelerometer_corrected;

    // Optional lever-arm compensation
    if (config_.enable_lever_arm_compensation && has_previous_angular_velocity_) {
        const Eigen::Vector3d angular_acceleration_wheel = 
            (angular_velocity_wheel - angular_velocity_wheel_previous_) / dt;
        acceleration_wheel = acceleration_wheel - 
            angular_acceleration_wheel.cross(translation_imu_in_wheel_) -
            angular_velocity_wheel.cross(angular_velocity_wheel.cross(translation_imu_in_wheel_));
    }

    // Cache angular velocity for next iteration
    angular_velocity_wheel_previous_ = angular_velocity_wheel;
    angular_velocity_wheel_last_ = angular_velocity_wheel;
    has_previous_angular_velocity_ = true;

    // Propagate nominal state
    const Eigen::Matrix3d rotation_world_T_wheel = nominal_state_.orientation.toRotationMatrix();
    const Eigen::Vector3d acceleration_world = 
        rotation_world_T_wheel * acceleration_wheel + config_.gravity_world;

    nominal_state_.position += nominal_state_.velocity * dt + 0.5 * acceleration_world * dt * dt;
    nominal_state_.velocity += acceleration_world * dt;
    nominal_state_.orientation = quaternionRightUpdate(nominal_state_.orientation, angular_velocity_wheel * dt);

    // Propagate error covariance
    propagateCovariance_(dt, acceleration_wheel, rotation_world_T_wheel);

    // Apply non-holonomic constraints
    applyNonHolonomicConstraints_();

    // Check for zero-velocity update opportunity
    tryTriggerInitializationOrZupt_(timestamp);
}

inline double ErrorStateKalmanFilter::updateWheelSpeed(double timestamp, double wheel_speed_raw) {
    wheel_last_scaled_speed_ = config_.wheel_speed_scale_factor * wheel_speed_raw;
    const bool is_wheel_static = std::abs(wheel_last_scaled_speed_) < config_.zupt_velocity_threshold;
    updateWheelStaticDetection_(timestamp, is_wheel_static);

    tryTriggerInitializationOrZupt_(timestamp);
    if (!is_initialized_) return 0.0;

    // Measurement model: forward velocity in wheel frame
    const Eigen::Matrix3d rotation_wheel_T_world = nominal_state_.orientation.conjugate().toRotationMatrix();
    const Eigen::Vector3d velocity_in_wheel_frame = rotation_wheel_T_world * nominal_state_.velocity;
    const double predicted_forward_velocity = velocity_in_wheel_frame.x();

    // Jacobian of measurement with respect to error state
    Eigen::Matrix<double, 1, ErrorState::STATE_DIMENSION> measurement_jacobian = 
        Eigen::Matrix<double, 1, ErrorState::STATE_DIMENSION>::Zero();
    
    // Derivative with respect to velocity error
    measurement_jacobian.block<1, 3>(0, 3) = Eigen::RowVector3d(1, 0, 0) * rotation_wheel_T_world;
    
    // Derivative with respect to orientation error
    measurement_jacobian.block<1, 3>(0, 6) = Eigen::RowVector3d(1, 0, 0) * skewSymmetric(velocity_in_wheel_frame);

    // Kalman filter update
    const double measurement_noise_variance = config_.wheel_speed_noise_std * config_.wheel_speed_noise_std;
    const double innovation_variance = (measurement_jacobian * error_state_.covariance * 
                                      measurement_jacobian.transpose())(0, 0) + measurement_noise_variance;
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, 1> kalman_gain = 
        error_state_.covariance * measurement_jacobian.transpose() / innovation_variance;
    
    const double innovation = wheel_last_scaled_speed_ - predicted_forward_velocity;
    error_state_.vector += kalman_gain * innovation;
    error_state_.covariance = (Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION>::Identity() - 
                             kalman_gain * measurement_jacobian) * error_state_.covariance;

    injectAndResetErrorState_();
    return innovation;
}

inline double ErrorStateKalmanFilter::updateWheelSpeed(double wheel_speed_raw) {
    return updateWheelSpeed(nominal_state_.timestamp, wheel_speed_raw);
}

inline void ErrorStateKalmanFilter::updateGPS(const GpsMeasurement& gps_measurement) {
    std::cout << "GPS update is not yet implemented, gps_measurement: " << gps_measurement.position_world.transpose();
    // TODO: Implement GPS update
    // This would involve a position measurement update with the GPS position and covariance
}

// ---------------- Private Method Implementations ----------------

inline void ErrorStateKalmanFilter::propagateCovariance_(double time_step,
                                                          const Eigen::Vector3d& acceleration_wheel,
                                                          const Eigen::Matrix3d& rotation_world_T_wheel) {
    Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION> state_transition_matrix = 
        Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION>::Zero();
    Eigen::Matrix<double, ErrorState::STATE_DIMENSION, 12> noise_jacobian = 
        Eigen::Matrix<double, ErrorState::STATE_DIMENSION, 12>::Zero();
    const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();

    // State transition matrix
    state_transition_matrix.block<3, 3>(0, 3) = identity3;  // position/velocity coupling
    state_transition_matrix.block<3, 3>(3, 6) = -rotation_world_T_wheel * skewSymmetric(acceleration_wheel);
    state_transition_matrix.block<3, 3>(3, 12) = -rotation_world_T_wheel * rotation_wheel_T_imu_;
    state_transition_matrix.block<3, 3>(6, 9) = -rotation_wheel_T_imu_;

    // Noise jacobian (IMU noise in IMU frame)
    noise_jacobian.block<3, 3>(6, 0) = -rotation_wheel_T_imu_;
    noise_jacobian.block<3, 3>(3, 3) = -rotation_world_T_wheel * rotation_wheel_T_imu_;
    noise_jacobian.block<3, 3>(9, 6) = identity3;
    noise_jacobian.block<3, 3>(12, 9) = identity3;

    // Continuous-time noise covariance matrix
    Eigen::Matrix<double, 12, 12> continuous_noise_covariance = Eigen::Matrix<double, 12, 12>::Zero();
    continuous_noise_covariance.block<3, 3>(0, 0) = 
        (config_.gyroscope_noise_density * config_.gyroscope_noise_density) * identity3;
    continuous_noise_covariance.block<3, 3>(3, 3) = 
        (config_.accelerometer_noise_density * config_.accelerometer_noise_density) * identity3;
    continuous_noise_covariance.block<3, 3>(6, 6) = 
        (config_.gyroscope_random_walk * config_.gyroscope_random_walk) * identity3;
    continuous_noise_covariance.block<3, 3>(9, 9) = 
        (config_.accelerometer_random_walk * config_.accelerometer_random_walk) * identity3;

    // Discretize matrices
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION> discrete_state_transition = 
        Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION>::Identity() + 
        state_transition_matrix * time_step;
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION> discrete_noise_covariance = 
        noise_jacobian * continuous_noise_covariance * noise_jacobian.transpose() * time_step;

    // Propagate covariance
    error_state_.covariance = discrete_state_transition * error_state_.covariance * 
                             discrete_state_transition.transpose() + discrete_noise_covariance;
}

inline void ErrorStateKalmanFilter::injectAndResetErrorState_() {
    // Extract error state components
    const Eigen::Vector3d position_error = error_state_.vector.block<3, 1>(0, 0);
    const Eigen::Vector3d velocity_error = error_state_.vector.block<3, 1>(3, 0);
    const Eigen::Vector3d orientation_error = error_state_.vector.block<3, 1>(6, 0);
    const Eigen::Vector3d gyroscope_bias_error = error_state_.vector.block<3, 1>(9, 0);
    const Eigen::Vector3d accelerometer_bias_error = error_state_.vector.block<3, 1>(12, 0);

    // Inject errors into nominal state
    nominal_state_.position += position_error;
    nominal_state_.velocity += velocity_error;
    nominal_state_.orientation = quaternionRightUpdate(nominal_state_.orientation, orientation_error);
    nominal_state_.gyroscope_bias += gyroscope_bias_error;
    nominal_state_.accelerometer_bias += accelerometer_bias_error;

    // Reset error state to zero
    error_state_.vector.setZero();
}

inline void ErrorStateKalmanFilter::performStaticInitialization_() {
    Eigen::Vector3d acceleration_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyroscope_mean = Eigen::Vector3d::Zero();
    
    for (const auto& accel : initialization_acceleration_buffer_) {
        acceleration_mean += accel;
    }
    for (const auto& gyro : initialization_gyroscope_buffer_) {
        gyroscope_mean += gyro;
    }
    
    const double sample_count = std::max<size_t>(1, initialization_acceleration_buffer_.size());
    acceleration_mean /= sample_count;
    gyroscope_mean /= sample_count;

    // Estimate gyroscope bias
    nominal_state_.gyroscope_bias = gyroscope_mean;

    // Align gravity vector to estimate initial orientation
    const Eigen::Vector3d gravity_world_normalized = config_.gravity_world.normalized();
    const Eigen::Vector3d negative_acceleration_normalized = (-acceleration_mean).normalized();
    const Eigen::Quaterniond orientation_imu_T_world = 
        Eigen::Quaterniond::FromTwoVectors(negative_acceleration_normalized, gravity_world_normalized);
    const Eigen::Matrix3d rotation_world_T_imu = orientation_imu_T_world.toRotationMatrix().transpose();

    // Estimate accelerometer bias
    nominal_state_.accelerometer_bias = acceleration_mean + rotation_world_T_imu * config_.gravity_world;

    // Compute initial wheel frame orientation
    const Eigen::Matrix3d rotation_world_T_wheel = orientation_imu_T_world.toRotationMatrix() * rotation_imu_T_wheel_;
    nominal_state_.orientation = Eigen::Quaterniond(rotation_world_T_wheel).normalized();
    nominal_state_.velocity.setZero();
    nominal_state_.timestamp = (initialization_start_time_ < 0) ? 0.0 : initialization_start_time_;
    
    has_previous_angular_velocity_ = false;
}

inline void ErrorStateKalmanFilter::applyZeroVelocityUpdate_() {
    Eigen::Matrix<double, 3, ErrorState::STATE_DIMENSION> measurement_jacobian = 
        Eigen::Matrix<double, 3, ErrorState::STATE_DIMENSION>::Zero();
    measurement_jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    
    const Eigen::Matrix3d measurement_noise_covariance = 
        (config_.zupt_velocity_noise_std * config_.zupt_velocity_noise_std) * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d innovation_covariance = 
        measurement_jacobian * error_state_.covariance * measurement_jacobian.transpose() + measurement_noise_covariance;
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, 3> kalman_gain = 
        error_state_.covariance * measurement_jacobian.transpose() * innovation_covariance.inverse();
    
    const Eigen::Vector3d innovation = -nominal_state_.velocity;
    error_state_.vector += kalman_gain * innovation;
    error_state_.covariance = (Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION>::Identity() - 
                             kalman_gain * measurement_jacobian) * error_state_.covariance;
    
    injectAndResetErrorState_();
}

inline void ErrorStateKalmanFilter::nudgeBiasesDuringStaticPeriod_() {
    if (zupt_acceleration_buffer_.empty() || zupt_gyroscope_buffer_.empty()) {
        return;
    }
    
    Eigen::Vector3d acceleration_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyroscope_mean = Eigen::Vector3d::Zero();
    
    for (const auto& accel : zupt_acceleration_buffer_) {
        acceleration_mean += accel;
    }
    for (const auto& gyro : zupt_gyroscope_buffer_) {
        gyroscope_mean += gyro;
    }
    
    acceleration_mean /= static_cast<double>(zupt_acceleration_buffer_.size());
    gyroscope_mean /= static_cast<double>(zupt_gyroscope_buffer_.size());

    // Apply exponential smoothing to bias estimates
    const double alpha = config_.zupt_bias_nudging_factor;
    nominal_state_.gyroscope_bias = (1.0 - alpha) * nominal_state_.gyroscope_bias + alpha * gyroscope_mean;
    
    const Eigen::Vector3d gravity_world_normalized = config_.gravity_world.normalized();
    const Eigen::Vector3d negative_acceleration_normalized = (-acceleration_mean).normalized();
    const Eigen::Quaterniond orientation_imu_T_world = 
        Eigen::Quaterniond::FromTwoVectors(negative_acceleration_normalized, gravity_world_normalized);
    const Eigen::Matrix3d rotation_world_T_imu = orientation_imu_T_world.toRotationMatrix().transpose();
    const Eigen::Vector3d accelerometer_bias_estimate = acceleration_mean + rotation_world_T_imu * config_.gravity_world;
    
    nominal_state_.accelerometer_bias = (1.0 - alpha) * nominal_state_.accelerometer_bias + alpha * accelerometer_bias_estimate;
}

inline void ErrorStateKalmanFilter::applyNonHolonomicConstraints_() {
    if (!is_initialized_ || nominal_state_.velocity.norm() < config_.nhc_velocity_threshold) {
        return;
    }

    const Eigen::Matrix3d rotation_wheel_T_world = nominal_state_.orientation.conjugate().toRotationMatrix();
    const Eigen::Vector3d velocity_in_wheel_frame = rotation_wheel_T_world * nominal_state_.velocity;

    // Measurement model: lateral and vertical velocities should be zero
    Eigen::Vector2d predicted_lateral_velocities;
    predicted_lateral_velocities << velocity_in_wheel_frame.y(), velocity_in_wheel_frame.z();
    const Eigen::Vector2d innovation = -predicted_lateral_velocities;

    // Jacobian matrix
    Eigen::Matrix<double, 2, ErrorState::STATE_DIMENSION> measurement_jacobian = 
        Eigen::Matrix<double, 2, ErrorState::STATE_DIMENSION>::Zero();
    
    measurement_jacobian.block<1, 3>(0, 3) = Eigen::RowVector3d(0, 1, 0) * rotation_wheel_T_world;
    measurement_jacobian.block<1, 3>(1, 3) = Eigen::RowVector3d(0, 0, 1) * rotation_wheel_T_world;
    measurement_jacobian.block<1, 3>(0, 6) = Eigen::RowVector3d(0, 1, 0) * skewSymmetric(velocity_in_wheel_frame);
    measurement_jacobian.block<1, 3>(1, 6) = Eigen::RowVector3d(0, 0, 1) * skewSymmetric(velocity_in_wheel_frame);

    // Kalman filter update
    const Eigen::Matrix2d measurement_noise_covariance = 
        Eigen::Matrix2d::Identity() * (config_.nhc_lateral_noise_std * config_.nhc_lateral_noise_std);
    const Eigen::Matrix2d innovation_covariance = 
        measurement_jacobian * error_state_.covariance * measurement_jacobian.transpose() + measurement_noise_covariance;
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, 2> kalman_gain = 
        error_state_.covariance * measurement_jacobian.transpose() * innovation_covariance.inverse();

    error_state_.vector += kalman_gain * innovation;
    
    // Joseph form update for numerical stability
    const Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION> identity_minus_kh = 
        Eigen::Matrix<double, ErrorState::STATE_DIMENSION, ErrorState::STATE_DIMENSION>::Identity() - kalman_gain * measurement_jacobian;
    error_state_.covariance = identity_minus_kh * error_state_.covariance * identity_minus_kh.transpose() + 
                             kalman_gain * measurement_noise_covariance * kalman_gain.transpose();

    injectAndResetErrorState_();
}

inline void ErrorStateKalmanFilter::updateImuStaticDetection_(double timestamp,
                                                              const Eigen::Vector3d& gyroscope_raw,
                                                              const Eigen::Vector3d& accelerometer_raw) {
    const double gravity_magnitude = config_.gravity_world.norm();
    const bool is_imu_static = (gyroscope_raw.norm() < config_.zupt_gyroscope_threshold) &&
                               (std::abs(accelerometer_raw.norm() - gravity_magnitude) < config_.zupt_acceleration_threshold);
    
    is_imu_static_ = is_imu_static;
    imu_last_timestamp_ = timestamp;

    const bool is_wheel_data_fresh = (wheel_last_timestamp_ >= 0.0) && 
                                    ((timestamp - wheel_last_timestamp_) <= config_.wheel_data_timeout);
    const bool both_sensors_static = is_imu_static && is_wheel_static_ && is_wheel_data_fresh;
    
    if (both_sensors_static) {
        if (static_window_start_time_ < 0) {
            static_window_start_time_ = std::max(timestamp, wheel_last_timestamp_);
        }
        ++static_window_sample_count_;
        zupt_acceleration_buffer_.push_back(accelerometer_raw);
        zupt_gyroscope_buffer_.push_back(gyroscope_raw);
    } else {
        clearStaticWindow_();
    }

    if (is_imu_static) {
        if (initialization_start_time_ < 0) {
            initialization_start_time_ = timestamp;
        }
        initialization_acceleration_buffer_.push_back(accelerometer_raw);
        initialization_gyroscope_buffer_.push_back(gyroscope_raw);
    }
}

inline void ErrorStateKalmanFilter::updateWheelStaticDetection_(double timestamp, bool is_wheel_static) {
    is_wheel_static_ = is_wheel_static;
    wheel_last_timestamp_ = timestamp;

    const bool is_imu_data_fresh = (imu_last_timestamp_ >= 0.0) && 
                                  ((timestamp - imu_last_timestamp_) <= config_.wheel_data_timeout);
    const bool both_sensors_static = is_wheel_static && is_imu_static_ && is_imu_data_fresh;
    
    if (both_sensors_static) {
        if (static_window_start_time_ < 0) {
            static_window_start_time_ = std::max(timestamp, imu_last_timestamp_);
        }
        ++static_window_sample_count_;
    } else {
        clearStaticWindow_();
    }
}

inline void ErrorStateKalmanFilter::tryTriggerInitializationOrZupt_(double current_timestamp) {
    if (static_window_start_time_ < 0) {
        return;
    }
    
    const bool duration_satisfied = (current_timestamp - static_window_start_time_) >= config_.zupt_minimum_duration;
    const bool sample_count_satisfied = static_window_sample_count_ >= config_.zupt_minimum_samples;
    
    if (!(duration_satisfied || sample_count_satisfied)) {
        return;
    }

    if (!is_initialized_) {
        performStaticInitialization_();
        is_initialized_ = true;
        is_waiting_for_init_ = false;
        clearInitializationBuffer_();
        clearStaticWindow_();
    } else {
        applyZeroVelocityUpdate_();
        nudgeBiasesDuringStaticPeriod_();
        clearStaticWindow_();
    }
}

inline void ErrorStateKalmanFilter::clearInitializationBuffer_() {
    initialization_acceleration_buffer_.clear();
    initialization_gyroscope_buffer_.clear();
    initialization_start_time_ = -1.0;
}

inline void ErrorStateKalmanFilter::clearStaticWindow_() {
    static_window_start_time_ = -1.0;
    static_window_sample_count_ = 0;
    zupt_acceleration_buffer_.clear();
    zupt_gyroscope_buffer_.clear();
}

} // namespace eskf
