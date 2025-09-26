#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <limits>
#include <yaml-cpp/yaml.h>
#include <string>

// Minimal Error-State Kalman Filter (ESKF) for a differential-drive robot.
// Sensors: IMU (acc, gyro) and wheel forward speed (no yaw-rate).
// Frames:
//   - IMU frame == body frame (b / i)
//   - World frame (w): defined at static initialization s.t. gravity is [0,0,-g].
// State (nominal): p_wi, v_wi, q_wi, b_a, b_g
//   p_wi: IMU position in world
//   v_wi: IMU velocity in world
//   q_wi: rotation_world_T_imu (maps imu vectors -> world)
//   b_a, b_g: accelerometer and gyro biases in IMU frame
// Error-state: [dp, dv, dtheta, dba, dbg] (15x1)
// Right-multiplicative quaternion update: q_new = q * Exp(dtheta)
// Measurement: wheel-frame velocity z = [vx, 0, 0]^T (in wheel frame).
// Known extrinsic: Transform_imu_T_wheel (R_iw, t_iw), mapping wheel->imu.
//   R_iw : Rotation_imu_T_wheel (Eigen::Quaterniond)
//   t_iw : position of wheel origin expressed in imu frame (vector from IMU origin to wheel origin)
// Prediction uses IMU mid-point integration (single-iteration midpoint).

class eskf {
public:
  struct Params {
    double sigma_acc = 0.8;             // m/s^2 / sqrt(Hz)
    double sigma_gyro = 0.02;            // rad/s / sqrt(Hz)
    double sigma_ba = 0.0005;            // m/s^2 / sqrt(Hz) (bias RW)
    double sigma_bg = 0.0002;            // rad/s / sqrt(Hz) (bias RW)
    double sigma_wheel_vx = 0.05;        // m/s (forward)
    double sigma_wheel_plane = 0.02;     // m/s (lateral & vertical constraints)
    double gravity = 9.81;               // m/s^2

    // Static initialization params
    int    init_min_samples = 200;       // ~2s at 100 Hz
    double init_max_gyro = 0.05;         // rad/s
    double init_acc_std_thresh = 0.1;    // m/s^2

    // Simple zero-velocity updates (ZUPT) when detected static
    bool   use_zupt = false;
    double zupt_max_gyro = 0.08;         // rad/s
    double zupt_acc_norm_thresh = 0.15;  // | |a| - g |
    double zupt_sigma_v = 0.02;          // m/s

    // Robust gating (chi-square) to avoid blow-ups
    double gate_chi2_wheel = 25.0;     // ~95% in 3D
    double gate_chi2_zupt  = 25.0;

    // 从YAML节点加载参数
    static Params fromYaml(const YAML::Node& node) {
      Params params;
      
      if (node["noise"]) {
        const auto& noise = node["noise"];
        if (noise["sigma_acc"]) params.sigma_acc = noise["sigma_acc"].as<double>();
        if (noise["sigma_gyro"]) params.sigma_gyro = noise["sigma_gyro"].as<double>();
        if (noise["sigma_ba"]) params.sigma_ba = noise["sigma_ba"].as<double>();
        if (noise["sigma_bg"]) params.sigma_bg = noise["sigma_bg"].as<double>();
        if (noise["sigma_wheel_vx"]) params.sigma_wheel_vx = noise["sigma_wheel_vx"].as<double>();
        if (noise["sigma_wheel_plane"]) params.sigma_wheel_plane = noise["sigma_wheel_plane"].as<double>();
      }
      
      if (node["gravity"]) {
        params.gravity = node["gravity"].as<double>();
      }
      
      if (node["init"]) {
        const auto& init = node["init"]; 
        if (init["min_samples"]) params.init_min_samples = init["min_samples"].as<int>();
        if (init["max_gyro"]) params.init_max_gyro = init["max_gyro"].as<double>();
        if (init["acc_std_thresh"]) params.init_acc_std_thresh = init["acc_std_thresh"].as<double>();
      }

      if (node["zupt"]) {
        const auto& z = node["zupt"]; 
        if (z["enable"]) params.use_zupt = z["enable"].as<bool>();
        if (z["max_gyro"]) params.zupt_max_gyro = z["max_gyro"].as<double>();
        if (z["acc_norm_thresh"]) params.zupt_acc_norm_thresh = z["acc_norm_thresh"].as<double>();
        if (z["sigma_v"]) params.zupt_sigma_v = z["sigma_v"].as<double>();
      }
      
      return params;
    }
  };

  struct NominalState {
    Eigen::Vector3d p = Eigen::Vector3d::Zero(); // world
    Eigen::Vector3d v = Eigen::Vector3d::Zero(); // world
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // rotation_world_T_imu
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    double timestamp = std::numeric_limits<double>::quiet_NaN();
    bool initialized = false;
  };

  // Constructor with known extrinsics: Rotation_imu_T_wheel (wheel->imu) and translation (wheel origin in imu frame)
  eskf(const Eigen::Quaterniond& Rotation_imu_T_wheel,
       const Eigen::Vector3d& t_imu_T_wheel,
       const Params& params)
  : R_iw_(Rotation_imu_T_wheel.normalized()), t_iw_i_(t_imu_T_wheel), prm_(params)
  {
    P_.setIdentity();
    P_ *= 1e-2; // modest initial uncertainty; will be reset at init

    Qc_.setZero();
    Qc_.block<3,3>(0,0) = (prm_.sigma_acc * prm_.sigma_acc) * Eigen::Matrix3d::Identity(); // n_a
    Qc_.block<3,3>(3,3) = (prm_.sigma_gyro * prm_.sigma_gyro) * Eigen::Matrix3d::Identity(); // n_g
    Qc_.block<3,3>(6,6) = (prm_.sigma_ba * prm_.sigma_ba) * Eigen::Matrix3d::Identity();   // n_ba
    Qc_.block<3,3>(9,9) = (prm_.sigma_bg * prm_.sigma_bg) * Eigen::Matrix3d::Identity();   // n_bg
  }

  // Constructor with default parameters
  eskf(const Eigen::Quaterniond& Rotation_imu_T_wheel,
       const Eigen::Vector3d& t_imu_T_wheel)
  : eskf(Rotation_imu_T_wheel, t_imu_T_wheel, Params{})
  {}

  // Feed one IMU sample (t in seconds; acc,gyro in IMU frame)
  void feedimu(double t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
    if (!has_imu_) {
      last_acc_ = acc;
      last_gyro_ = gyro;
      last_t_ = t;
      has_imu_ = true;
    }

    // Static initialization window
    if (!state_.initialized) {
      init_buffer_.push_back({t, acc, gyro});
      if ((int)init_buffer_.size() > prm_.init_min_samples) {
        try_static_initialize();
        // drop old samples to keep buffer light
        if ((int)init_buffer_.size() > 3 * prm_.init_min_samples) {
          init_buffer_.erase(init_buffer_.begin(), init_buffer_.end() - prm_.init_min_samples);
        }
      }
      // Keep updating last measurement even before init
      last_acc_ = acc; last_gyro_ = gyro; last_t_ = t; 
      return;
    }

    if (t <= last_t_) {
      // Non-increasing timestamp, ignore
      last_acc_ = acc; last_gyro_ = gyro; last_t_ = t; 
      return;
    }

    // Mid-point propagate from (last_t_, last_acc_, last_gyro_) to (t, acc, gyro)
    const double dt = t - last_t_;
    midpoint_propagate(dt, last_acc_, acc, last_gyro_, gyro);

    // Covariance propagation
    const Eigen::Vector3d a_avg = 0.5*(last_acc_ + acc) - state_.ba;
    const Eigen::Vector3d w_avg = 0.5*(last_gyro_ + gyro) - state_.bg;
    imu_prop_cov(dt, a_avg, w_avg);

    last_acc_ = acc;
    last_gyro_ = gyro;
    last_t_ = t;
    state_.timestamp = t;

    // Optional ZUPT to clamp drift when nearly static
    maybe_zupt_update(0.5*(last_acc_ + acc), 0.5*(last_gyro_ + gyro));
  }

  // Feed wheel forward speed (m/s). z = [vx, 0, 0] in wheel frame.
  // If desired, you can call this at a lower rate than IMU.
  void feedwheelvelocity(double t, double vx_wheel) {
    if (!has_imu_) return; // no timing reference yet

    // If we already have an initialized state and the wheel message is ahead of last_t_,
    // perform a simple propagate using constant last IMU sample to align time.
    if (state_.initialized && t > last_t_) {
      const double dt = t - last_t_;
      // Use hold last sample as both ends for a zero-order midpoint step
      midpoint_propagate(dt, last_acc_, last_acc_, last_gyro_, last_gyro_);
      imu_prop_cov(dt, last_acc_ - state_.ba, last_gyro_ - state_.bg);
      last_t_ = t;
      state_.timestamp = t;
    }

    if (!state_.initialized) return; // still initializing

    // Predicted velocity at wheel origin, expressed in wheel frame
    // Simplified kinematics: h = R_wi * ( R_i^w * v_w + s_i ), where
    //   R_wi : wheel<-imu
    //   R_i^w: imu<-world = q^*
    //   s_i  : (omega_i x r_PO)_i (all in IMU frame)
    const Eigen::Matrix3d R_wi = R_iw_.toRotationMatrix().transpose(); // wheel<-imu
    const Eigen::Matrix3d R_iTworld = state_.q.conjugate().toRotationMatrix(); // imu<-world

    const Eigen::Vector3d r_PO_i = t_iw_i_;                   // imu->wheel, expressed in imu
    const Eigen::Vector3d omega_i = last_gyro_ - state_.bg;   // imu angular velocity
    const Eigen::Vector3d s_i = omega_i.cross(r_PO_i);        // imu frame

    const Eigen::Vector3d v_i = R_iTworld * state_.v;         // IMU linear velocity expressed in IMU
    const Eigen::Vector3d h = R_wi * ( v_i + s_i );           // wheel frame predicted measurement

    // Measurement z = [vx, 0, 0]^T
    Eigen::Vector3d z; z << vx_wheel, 0.0, 0.0;
    const Eigen::Vector3d y = z - h; // residual in wheel frame

    // Jacobian H (3x15): dv, dtheta, dbg
    Eigen::Matrix<double,3,15> H; H.setZero();
    // dv term: ∂h/∂v = R_wi * R_i^w
    H.block<3,3>(0,3) = R_wi * R_iTworld;
    // dtheta term (right-mult): R_i^w -> Exp(-dθ) R_i^w, so δ(R_i^w v) = - [v_i]_x dθ
    H.block<3,3>(0,6) = R_wi * ( - skew(v_i) );
    // dbg term via s_i = (ω - bg) x r ⇒ ∂s/∂bg = - [r]_x
    H.block<3,3>(0,12) = R_wi * ( - skew(r_PO_i) );

    // Measurement noise
    Eigen::Matrix3d Rm = Eigen::Matrix3d::Zero();
    Rm(0,0) = prm_.sigma_wheel_vx * prm_.sigma_wheel_vx;
    Rm(1,1) = prm_.sigma_wheel_plane * prm_.sigma_wheel_plane;
    Rm(2,2) = prm_.sigma_wheel_plane * prm_.sigma_wheel_plane;

    // Kalman gain and update (LDLT + Joseph + gating)
    const Eigen::Matrix3d S = (H * P_ * H.transpose()) + Rm;
    Eigen::LDLT<Eigen::Matrix3d> Sldlt(S);
    if (Sldlt.info() != Eigen::Success) return; // numerical safeguard

    const Eigen::Vector3d Sinv_y = Sldlt.solve(y);
    const double gamma = y.dot(Sinv_y);
    if (gamma > prm_.gate_chi2_wheel) return; // outlier reject

    const Eigen::Matrix3d Sinv = Sldlt.solve(Eigen::Matrix3d::Identity());
    const Eigen::Matrix<double,15,3> K = P_ * H.transpose() * Sinv;
    const Eigen::Matrix<double,15,1> dx = K * y;

    apply_error_state(dx);

    const Eigen::Matrix<double,15,15> I15 = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I15 - K * H) * P_ * (I15 - K * H).transpose() + K * Rm * K.transpose(); // Joseph form
  }

  NominalState getNominalState() const { return state_; }

  // 从YAML配置文件创建eskf实例的静态函数
  static eskf fromConfigFile(const std::string& config_path, Eigen::Quaterniond& R_iw, Eigen::Vector3d& t_iw) {
    YAML::Node config = YAML::LoadFile(config_path);
    
    if (!config["eskf"]) {
      throw std::runtime_error("ESKF configuration not found in config file");
    }
    
    const auto& eskf_config = config["eskf"];
    
    // 加载参数
    Params params = Params::fromYaml(eskf_config);
    
    return eskf(R_iw, t_iw, params);
  }

private:
  struct ImuSample { double t; Eigen::Vector3d a; Eigen::Vector3d g; };

  void maybe_zupt_update(const Eigen::Vector3d& acc_i, const Eigen::Vector3d& gyro_i) {
    if (!state_.initialized || !prm_.use_zupt) return;
    const double gyro_norm = gyro_i.norm();
    const double acc_norm_err = std::abs(acc_i.norm() - prm_.gravity);
    if (gyro_norm > prm_.zupt_max_gyro || acc_norm_err > prm_.zupt_acc_norm_thresh) return;

    // z = 0 - v_world
    Eigen::Matrix<double,3,15> H; H.setZero();
    H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d y = - state_.v;
    const Eigen::Matrix3d Rv = (prm_.zupt_sigma_v * prm_.zupt_sigma_v) * Eigen::Matrix3d::Identity();

    const Eigen::Matrix3d S = (H * P_ * H.transpose()) + Rv;
    Eigen::LDLT<Eigen::Matrix3d> Sldlt(S);
    if (Sldlt.info() != Eigen::Success) return;

    const Eigen::Vector3d Sinv_y = Sldlt.solve(y);
    const double gamma = y.dot(Sinv_y);
    if (gamma > prm_.gate_chi2_zupt) return;

    const Eigen::Matrix3d Sinv = Sldlt.solve(Eigen::Matrix3d::Identity());
    const Eigen::Matrix<double,15,3> K = P_ * H.transpose() * Sinv;
    const Eigen::Matrix<double,15,1> dx = K * y;

    apply_error_state(dx);

    const Eigen::Matrix<double,15,15> I15 = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I15 - K * H) * P_ * (I15 - K * H).transpose() + K * Rv * K.transpose();
  }

  static inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m; m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0; return m;
  }

  static inline Eigen::Quaterniond ExpSO3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    Eigen::Quaterniond dq;
    if (theta < 1e-8) {
      dq.w() = 1.0;
      dq.vec() = 0.5 * w; // first-order
    } else {
      double half = 0.5 * theta;
      double s = std::sin(half) / theta;
      dq.w() = std::cos(half);
      dq.vec() = s * w;
    }
    return dq.normalized();
  }

  static inline Eigen::Quaterniond quat_from_two_vectors(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    // returns q such that q*a = b (both in same frame). a,b should be normalized.
    Eigen::Vector3d va = a.normalized();
    Eigen::Vector3d vb = b.normalized();
    double c = va.dot(vb);
    if (c < -0.999999) {
      // 180 deg: pick arbitrary orthogonal axis
      Eigen::Vector3d axis = va.unitOrthogonal();
      return Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, axis));
    }
    Eigen::Vector3d axis = va.cross(vb);
    Eigen::Quaterniond q(1.0 + c, axis.x(), axis.y(), axis.z());
    q.normalize();
    return q;
  }

  void try_static_initialize() {
    if ((int)init_buffer_.size() < prm_.init_min_samples) return;

    // Use the last init_min_samples window
    const int N = prm_.init_min_samples;
    double t0 = init_buffer_.back().t;
    size_t start = init_buffer_.size() - N;

    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    for (size_t i = start; i < init_buffer_.size(); ++i) {
      acc_mean += init_buffer_[i].a;
      gyro_mean += init_buffer_[i].g;
    }
    acc_mean /= (double)N;
    gyro_mean /= (double)N;

    // Check stillness: small gyro and low acc scatter around |g|
    Eigen::Vector3d acc_var = Eigen::Vector3d::Zero();
    for (size_t i = start; i < init_buffer_.size(); ++i) {
      Eigen::Vector3d da = init_buffer_[i].a - acc_mean;
      acc_var += da.cwiseProduct(da);
    }
    acc_var /= (double)N;

    if (gyro_mean.norm() > prm_.init_max_gyro) return;
    if (acc_var.cwiseSqrt().maxCoeff() > prm_.init_acc_std_thresh) return;

    // Initial orientation: use specific force direction to align +Z (up) so that R_wi * a_mean ≈ +g_up.
    // At rest: f_b ≈ R^T ( -g_w ), with g_w = [0,0,-g]. Therefore R * f_b ≈ +[0,0,g]/g → +Z.
    const Eigen::Vector3d z_up(0,0,1);
    Eigen::Quaterniond q_wi = quat_from_two_vectors(acc_mean.normalized(), z_up);

    state_.q = q_wi.normalized();
    state_.v.setZero();
    state_.p.setZero();
    state_.bg = gyro_mean; // gyro bias = mean gyro at rest
    // initialize accelerometer bias so that predicted specific force matches measurement at rest
    state_.ba = acc_mean - state_.q.conjugate().toRotationMatrix() * Eigen::Vector3d(0,0,prm_.gravity);
    state_.timestamp = t0;
    state_.initialized = true;

    // Reset covariance to be confident in orientation tilt but unsure yaw
    P_.setZero();
    P_.block<3,3>(0,0) = 1e-2 * Eigen::Matrix3d::Identity(); // p
    P_.block<3,3>(3,3) = 1e-2 * Eigen::Matrix3d::Identity(); // v
    // Orientation: small roll/pitch, larger yaw
    Eigen::Vector3d ori_sigma(5e-3, 5e-3, 5e-2);
    P_.block<3,3>(6,6) = ori_sigma.cwiseProduct(ori_sigma).asDiagonal();
    P_.block<3,3>(9,9) = (5e-3 * 5e-3) * Eigen::Matrix3d::Identity(); // ba
    P_.block<3,3>(12,12) = (5e-4 * 5e-4) * Eigen::Matrix3d::Identity(); // bg
  }

  void midpoint_propagate(double dt,
                          const Eigen::Vector3d& a_k,
                          const Eigen::Vector3d& a_k1,
                          const Eigen::Vector3d& w_k,
                          const Eigen::Vector3d& w_k1) {
    if (dt <= 0) return;

    const Eigen::Vector3d w_unbias_avg = 0.5*(w_k + w_k1) - state_.bg;
    const Eigen::Quaterniond dq = ExpSO3(w_unbias_avg * dt);
    const Eigen::Quaterniond q_k1 = (state_.q * dq).normalized();

    const Eigen::Vector3d a_unbias_k  = a_k  - state_.ba;
    const Eigen::Vector3d a_unbias_k1 = a_k1 - state_.ba;

    const Eigen::Vector3d acc_world_k  = state_.q.toRotationMatrix() * a_unbias_k + Eigen::Vector3d(0,0,-prm_.gravity);
    const Eigen::Vector3d acc_world_k1 = q_k1.toRotationMatrix() * a_unbias_k1 + Eigen::Vector3d(0,0,-prm_.gravity);
    const Eigen::Vector3d acc_world_mid = 0.5 * (acc_world_k + acc_world_k1);

    // integrate
    state_.p += state_.v * dt + 0.5 * acc_world_mid * dt * dt;
    state_.v += acc_world_mid * dt;
    state_.q = q_k1;
    state_.q.normalize();
  }

  void imu_prop_cov(double dt, const Eigen::Vector3d& a_unbias_avg_i, const Eigen::Vector3d& w_unbias_avg_i) {
    // Build F and G at mid-point
    const Eigen::Matrix3d R_w = state_.q.toRotationMatrix(); // world<-imu

    Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
    F.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    F.block<3,3>(3,6) = - R_w * skew(a_unbias_avg_i); // dv/dtheta
    F.block<3,3>(3,9) = - R_w;                        // dv/dba
    F.block<3,3>(6,6) = - skew(w_unbias_avg_i);       // dtheta/dtheta
    F.block<3,3>(6,12)= - Eigen::Matrix3d::Identity(); // dtheta/dbg

    // Noise mapping G (15x12) for [n_a, n_g, n_ba, n_bg]
    Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
    G.block<3,3>(3,0)  = R_w;                         // acc noise to dv
    G.block<3,3>(6,3)  = - Eigen::Matrix3d::Identity(); // gyro noise to dtheta
    G.block<3,3>(9,6)  = Eigen::Matrix3d::Identity();  // ba random walk
    G.block<3,3>(12,9) = Eigen::Matrix3d::Identity();  // bg random walk

    const Eigen::Matrix<double,15,15> Phi = Eigen::Matrix<double,15,15>::Identity() + F * dt; // 1st order
    const Eigen::Matrix<double,15,15> Qd = G * Qc_ * G.transpose() * dt;

    P_ = Phi * P_ * Phi.transpose() + Qd;
  }

  void apply_error_state(const Eigen::Matrix<double,15,1>& dx) {
    state_.p += dx.segment<3>(0);
    state_.v += dx.segment<3>(3);
    const Eigen::Vector3d dtheta = dx.segment<3>(6);
    state_.q = (state_.q * ExpSO3(dtheta)).normalized(); // right-mult
    state_.ba += dx.segment<3>(9);
    state_.bg += dx.segment<3>(12);

    // Minimal consistent reset for right-mult error: rotate orientation error subspace
    Eigen::Matrix3d J = Eigen::Matrix3d::Identity() - 0.5 * skew(dtheta);
    Eigen::Matrix<double,15,15> G = Eigen::Matrix<double,15,15>::Identity();
    G.block<3,3>(6,6) = J;
    P_ = G * P_ * G.transpose();
  }

private:
  // Known extrinsics
  Eigen::Quaterniond R_iw_;        // Rotation_imu_T_wheel (wheel->imu)
  Eigen::Vector3d    t_iw_i_;      // wheel origin in imu frame (vector from IMU origin to wheel origin)

  Params prm_;
  NominalState state_;

  // Covariance and continuous-time noise
  Eigen::Matrix<double,15,15> P_;
  Eigen::Matrix<double,12,12> Qc_;

  // IMU buffering / last sample
  bool has_imu_ = false;
  Eigen::Vector3d last_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero();
  double last_t_ = 0.0;

  std::deque<ImuSample> init_buffer_;
};
