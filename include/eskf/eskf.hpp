#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

/**
 * ESKF for differential-drive robot using WHEEL frame as the vehicle frame.
 *
 * Frames:
 *  - w: world/inertial, gravity g_w = [0,0,-9.81] (ENU convention)
 *  - wheel: vehicle frame located at differential center (the state pose is R_{wwheel}, p_w)
 *  - i: IMU sensor frame at arbitrary location on the robot
 *
 * Known rigid transform: T_wheel_imu (wheel <- imu), includes rotation R_wi and translation t_wi.
 * We rotate IMU measurements from i to wheel by R_wi, and (optionally) account for lever arm r_wi^wheel = t_wi
 * when compensating linear acceleration at wheel center.
 *
 * State (nominal): x = { p_w (3), v_w (3), q_wheel (Quaternion world<-wheel), bg_i (3), ba_i (3) }
 * Error-state: 15-dim [dp dv dtheta dbg_i dba_i]
 *
 * Process model: standard strapdown with IMU measurements (biases defined in i-frame).
 * Observation: scalar wheel forward speed along wheel-X: z = ex^T * R_wheel_world * v_w + noise
 *   (No omega x r term needed because the state origin is at the wheel center already.)
 *
 * Auto-initialization & ZUPT: when both IMU and wheel indicate static over a window, 
 *   - If not initialized: estimate bg_i, ba_i, align gravity to obtain q_wheel^0
 *   - If initialized: apply ZUPT (velocity=0) + slow bias nudging
 */

namespace eskf {

// ---------------- tools ----------------
inline Eigen::Matrix3d Skew(const Eigen::Vector3d& a) {
    Eigen::Matrix3d S;
    S <<     0, -a.z(),  a.y(),
          a.z(),     0, -a.x(),
         -a.y(),  a.x(),     0;
    return S;
}
inline Eigen::Quaterniond RightUpdate(const Eigen::Quaterniond& q, const Eigen::Vector3d& dtheta) {
    Eigen::Quaterniond dq(1, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
    return (q * dq).normalized();
}

// ---------------- config/state ----------------
struct Config {
    // gravity in world
    Eigen::Vector3d g_world = {0,0,-9.81};

    // continuous-time noise (PSD^0.5)
    double gyro_noise   = 1.5e-3;   // rad/s/sqrt(Hz)
    double accel_noise  = 2.5e-1;   // m/s^2/sqrt(Hz)
    double gyro_rw      = 1.0e-5;   // rad/s^2/sqrt(Hz)
    double accel_rw     = 1.0e-4;   // m/s^3/sqrt(Hz)

    // wheel speed obs
    double wheel_sigma  = 0.0001;     // m/s
    double wheel_scale  = 1.0;      // scale factor for wheel speed

    // known extrinsic: wheel <- imu
    Eigen::Isometry3d T_wheel_imu = Eigen::Isometry3d::Identity();

    // ZUPT/static detection
    double zupt_speed_th    = 0.03; // m/s
    double zupt_gyro_th     = 0.02; // rad/s
    double zupt_acc_norm_th = 0.06; // m/s^2
    double zupt_time_min    = 1.5;  // s
    size_t zupt_min_count   = 120;  // samples
    double wheel_stale_max  = 0.5;  // s
    double zupt_sigma_v     = 0.02; // m/s (velocity zero obs noise)
    double zupt_bias_alpha  = 0.03; // slow pullback for ba/bg in static

    // Optional lever-arm compensation (use with care; requires omega dot approx)
    bool   enable_lever_arm = false;
};

struct State {
    double t = 0.0;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();     // world
    Eigen::Vector3d v = Eigen::Vector3d::Zero();     // world
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // world<-wheel
    // biases in IMU frame i
    Eigen::Vector3d bg_i = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_i = Eigen::Vector3d::Zero();
};

struct ErrorState {
    Eigen::Matrix<double,15,1> x = Eigen::Matrix<double,15,1>::Zero();
    Eigen::Matrix<double,15,15> P = Eigen::Matrix<double,15,15>::Identity() * 1e-3;
};

// ---------------- ESKF ----------------
class ESKF {
public:
    explicit ESKF(const Config& cfg): cfg_(cfg) {
        R_wi_ = cfg_.T_wheel_imu.linear();
        R_iw_ = R_wi_.transpose();
        r_wi_in_wheel_ = cfg_.T_wheel_imu.translation(); // IMU position expressed in wheel frame
    }

    // basic access
    void setInitial(const State& s0, const ErrorState& e0=ErrorState()) {
        state_ = s0; err_ = e0; initialized_ = true; waiting_init_ = false;
        clearInitBuffer_(); clearStaticWindow_();
        omega_w_prev_valid_ = false;
    }
    bool initialized()   const { return initialized_; }
    bool waitingInit()   const { return waiting_init_; }
    const State& state() const { return state_; }
    const ErrorState& error() const { return err_; }

    void setWheelScale(double s){ cfg_.wheel_scale = s; }
    void setT_wheel_imu(const Eigen::Isometry3d& T) {
        cfg_.T_wheel_imu = T; R_wi_ = T.linear(); R_iw_ = R_wi_.transpose(); r_wi_in_wheel_ = T.translation();
    }

    // ------------- prediction (feed IMU raw in i-frame) -------------
    void predict(double t, const Eigen::Vector3d& w_i_raw, const Eigen::Vector3d& a_i_raw) {
        // Update IMU static window and buffers for init/zupt
        updateImuStatic_(t, w_i_raw, a_i_raw);

        if (!initialized_) { tryTriggerInitOrZUPT_(t); state_.t = t; return; }

        const double dt = std::max(1e-6, t - state_.t);
        state_.t = t;

        // 1) de-bias in i-frame
        const Eigen::Vector3d w_i = w_i_raw - state_.bg_i;
        const Eigen::Vector3d a_i = a_i_raw - state_.ba_i;

        // 2) rotate to wheel frame
        const Eigen::Vector3d w_wh = R_wi_ * w_i; // angular rate expressed in wheel frame
        Eigen::Vector3d a_wh = R_wi_ * a_i;       // specific force in wheel frame at IMU location
        std::cout << "a_wh: " << a_wh.transpose() << std::endl;

        // Optional lever-arm compensation: transfer acceleration from IMU point to wheel origin
        if (cfg_.enable_lever_arm) {
            // a_origin = a_point - (alpha x r) - (omega x (omega x r)), in wheel frame
            // Use finite difference for alpha (angular acceleration) if previous omega exists
            if (omega_w_prev_valid_) {
                const Eigen::Vector3d alpha_w = (w_wh - omega_w_prev_) / dt;
                a_wh = a_wh - alpha_w.cross(r_wi_in_wheel_) - w_wh.cross(w_wh.cross(r_wi_in_wheel_));
            }
        }
        omega_w_prev_ = w_wh; omega_w_prev_valid_ = true;
        omega_w_last_ = w_wh; // keep for potential debug

        // 3) propagate nominal state in world frame
        const Eigen::Matrix3d Rwwheel = state_.q.toRotationMatrix();
        const Eigen::Vector3d a_world = Rwwheel * a_wh + cfg_.g_world; // specific->accel
        std::cout << "a_world: " << a_world.transpose() << std::endl;
        std::cout << "--------------------------------" << std::endl;

        // state_.v += a_world * dt;
        // state_.p += state_.v * dt + 0.5 * a_world * dt * dt;
        state_.q  = RightUpdate(state_.q, w_wh * dt);

        // 4) covariance propagation
        propagateCov_(dt, a_wh, Rwwheel);

        // try ZUPT if window satisfied
        tryTriggerInitOrZUPT_(t);
    }

    // ------------- wheel speed measurement (scalar, with timestamp) -------------
    // returns innovation (predicted - measured)
    double updateWheel(double t, double wheel_speed_raw) {
        // update wheel static window
        wheel_last_scaled_ = cfg_.wheel_scale * wheel_speed_raw;
        const bool wheel_static = std::abs(wheel_last_scaled_) < cfg_.zupt_speed_th;
        updateWheelStatic_(t, wheel_static);

        // initialization / ZUPT triggers if ready
        tryTriggerInitOrZUPT_(t);
        if (!initialized_) return 0.0;

        // measurement model: z = ex^T * R_wheel_world * v_w
        const Eigen::Matrix3d R_wheel_world = state_.q.conjugate().toRotationMatrix(); // wheel<-world
        const double h = (Eigen::RowVector3d(1,0,0) * R_wheel_world * state_.v)(0);
        const double z = wheel_last_scaled_;

        Eigen::Matrix<double,1,15> H; H.setZero();
        // dh/dv_w = ex^T * R_wheel_world
        H.block<1,3>(0,3) = Eigen::RowVector3d(1,0,0) * R_wheel_world;
        // dh/dtheta via numeric diff (small, often negligible, but keep robustness)
        const double eps = 1e-6;
        for (int k=0;k<3;++k) {
            Eigen::Vector3d d=Eigen::Vector3d::Zero(); d[k]=eps;
            auto f = [&](const Eigen::Quaterniond& q)->double{
                const Eigen::Matrix3d Rww = q.conjugate().toRotationMatrix();
                return (Eigen::RowVector3d(1,0,0) * Rww * state_.v)(0);
            };
            const double hp = f(RightUpdate(state_.q, d));
            const double hm = f(RightUpdate(state_.q,-d));
            H(0,6+k) = (hp - hm)/(2.0*eps);
        }

        const double Rm = cfg_.wheel_sigma * cfg_.wheel_sigma;
        const double S  = (H * err_.P * H.transpose())(0,0) + Rm;
        const Eigen::Matrix<double,15,1> K = err_.P * H.transpose() * (1.0/S);
        const double y = z - h;
        err_.x += K * y;
        err_.P  = (Eigen::Matrix<double,15,15>::Identity() - K*H) * err_.P;
        injectAndReset_();
        return (h - z);
    }

    // compatibility (no timestamp): use current state time
    double updateWheel(double wheel_speed_raw) { return updateWheel(state_.t, wheel_speed_raw); }

    // RTK placeholder
    struct RtkMeas { double t=0; Eigen::Vector3d pos_world=Eigen::Vector3d::Zero(); Eigen::Matrix3d cov=Eigen::Matrix3d::Identity()*0.25; bool has_pos=true; };
    void updateRTK(const RtkMeas&) { /* TODO */ }

private:
    // ---- covariance propagation ----
    void propagateCov_(double dt, const Eigen::Vector3d& a_wh, const Eigen::Matrix3d& Rwwheel) {
        Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
        const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

        // a_wh defined in wheel frame
        F.block<3,3>(0,3)  = I3;                         // dp/dv
        F.block<3,3>(3,6)  = - Rwwheel * Skew(a_wh);     // dv/dtheta
        F.block<3,3>(3,12) = - Rwwheel * R_wi_;          // dv/dba_i  (a_wh = R_wi*(a_i - ba_i))
        F.block<3,3>(6,9)  = - I3;                       // dtheta/dbg_i (w_wh = R_wi*(w_i - bg_i))

        // noise in i-frame: [n_gi, n_ai, n_wg_i, n_wa_i]
        G.block<3,3>(6,0)   = - R_wi_;                   // dtheta/n_gi
        G.block<3,3>(3,3)   = - Rwwheel * R_wi_;         // dv/n_ai
        G.block<3,3>(9,6)   =  I3;                       // dbg_i / n_wg_i
        G.block<3,3>(12,9)  =  I3;                       // dba_i / n_wa_i

        const double sg  = cfg_.gyro_noise;
        const double sa  = cfg_.accel_noise;
        const double swg = cfg_.gyro_rw;
        const double swa = cfg_.accel_rw;

        Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
        Qc.block<3,3>(0,0) = (sg*sg)  * I3;
        Qc.block<3,3>(3,3) = (sa*sa)  * I3;
        Qc.block<3,3>(6,6) = (swg*swg)* I3;
        Qc.block<3,3>(9,9) = (swa*swa)* I3;

        const Eigen::Matrix<double,15,15> Phi = Eigen::Matrix<double,15,15>::Identity() + F*dt;
        const Eigen::Matrix<double,15,15> Qd  = G * Qc * G.transpose() * dt;
        err_.P = Phi * err_.P * Phi.transpose() + Qd;
    }

    // ---- inject ----
    void injectAndReset_() {
        state_.p   += err_.x.block<3,1>(0,0);
        state_.v   += err_.x.block<3,1>(3,0);
        state_.q    = RightUpdate(state_.q, err_.x.block<3,1>(6,0));
        state_.bg_i += err_.x.block<3,1>(9,0);
        state_.ba_i += err_.x.block<3,1>(12,0);
        err_.x.setZero();
    }

    // ---- static initialization ----
    void doStaticInit_() {
        Eigen::Vector3d a_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d w_mean = Eigen::Vector3d::Zero();
        for (auto& a : init_acc_buf_) a_mean += a;
        for (auto& w : init_gyr_buf_) w_mean += w;
        const double N = std::max<size_t>(1, init_acc_buf_.size());
        a_mean /= N; w_mean /= N;

        // biases in i-frame
        state_.bg_i = w_mean;
        // Align gravity using accelerometer
        const Eigen::Vector3d z_w = cfg_.g_world.normalized();
        const Eigen::Vector3d z_i = (-a_mean).normalized();
        const Eigen::Quaterniond q_wi = Eigen::Quaterniond::FromTwoVectors(z_i, z_w);
        const Eigen::Matrix3d R_wi = q_wi.toRotationMatrix();
        const Eigen::Matrix3d R_iw = R_wi.transpose();
        state_.ba_i = a_mean + R_iw * cfg_.g_world; // a ≈ ba_i + R_iw*(-g)

        // initial wheel attitude: R_wheel = R_wi * R_iw (compose with known R_wi? No: here q_wi maps i->w; we want world<-wheel
        // We know wheel<-imu = R_wi_ ; imu<-wheel = R_iw_. So world<-wheel = (world<-imu)*(imu<-wheel) = R_wi * R_iw_
        const Eigen::Matrix3d R_wheel = R_wi * R_iw_;
        state_.q = Eigen::Quaterniond(R_wheel).normalized();
        state_.v.setZero();
        state_.t = (init_t_first_ < 0) ? 0.0 : init_t_first_;
        omega_w_prev_valid_ = false; // reset derivative
    }

    // ---- ZUPT velocity zero ----
    void applyZUPT_() {
        Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
        H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d Rm = (cfg_.zupt_sigma_v * cfg_.zupt_sigma_v) * Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d S  = H * err_.P * H.transpose() + Rm;
        const Eigen::Matrix<double,15,3> K = err_.P * H.transpose() * S.inverse();
        const Eigen::Vector3d y = - state_.v;
        err_.x += K * y;
        err_.P  = (Eigen::Matrix<double,15,15>::Identity() - K*H) * err_.P;
        injectAndReset_();
    }

    // ---- bias nudging during static ----
    void nudgeBiasOnStatic_() {
        if (zupt_acc_buf_.empty() || zupt_gyr_buf_.empty()) return;
        Eigen::Vector3d a_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d w_mean = Eigen::Vector3d::Zero();
        for (auto& a : zupt_acc_buf_) a_mean += a;
        for (auto& w : zupt_gyr_buf_) w_mean += w;
        a_mean /= double(zupt_acc_buf_.size());
        w_mean /= double(zupt_gyr_buf_.size());

        state_.bg_i = (1.0 - cfg_.zupt_bias_alpha) * state_.bg_i + cfg_.zupt_bias_alpha * w_mean;
        const Eigen::Vector3d z_w = cfg_.g_world.normalized();
        const Eigen::Vector3d z_i = (-a_mean).normalized();
        const Eigen::Quaterniond q_wi = Eigen::Quaterniond::FromTwoVectors(z_i, z_w);
        const Eigen::Matrix3d R_iw = q_wi.toRotationMatrix().transpose();
        const Eigen::Vector3d ba_hat = a_mean + R_iw * cfg_.g_world;
        state_.ba_i = (1.0 - cfg_.zupt_bias_alpha) * state_.ba_i + cfg_.zupt_bias_alpha * ba_hat;
    }

    // ---- static windows & triggers ----
    void updateImuStatic_(double t, const Eigen::Vector3d& w_i_raw, const Eigen::Vector3d& a_i_raw) {
        const double g = cfg_.g_world.norm();
        const bool imu_static = (w_i_raw.norm() < cfg_.zupt_gyro_th) &&
                                (std::abs(a_i_raw.norm() - g) < cfg_.zupt_acc_norm_th);
        imu_static_last_ = imu_static; imu_last_t_ = t;

        const bool wheel_fresh = (wheel_last_t_ >= 0.0) && ((t - wheel_last_t_) <= cfg_.wheel_stale_max);
        const bool both_static = imu_static && wheel_static_last_ && wheel_fresh;
        if (both_static) {
            if (static_win_start_t_ < 0) static_win_start_t_ = std::max(t, wheel_last_t_);
            ++static_win_count_;
            // buffers
            zupt_acc_buf_.push_back(a_i_raw);
            zupt_gyr_buf_.push_back(w_i_raw);
        } else {
            clearStaticWindow_();
        }

        if (imu_static) { if (init_t_first_ < 0) init_t_first_ = t; init_acc_buf_.push_back(a_i_raw); init_gyr_buf_.push_back(w_i_raw); }
    }

    void updateWheelStatic_(double t, bool wheel_static) {
        wheel_static_last_ = wheel_static; wheel_last_t_ = t;
        const bool imu_fresh = (imu_last_t_ >= 0.0);
        const bool both_static = wheel_static && imu_static_last_ && imu_fresh && ((t - wheel_last_t_) <= cfg_.wheel_stale_max);
        if (both_static) {
            if (static_win_start_t_ < 0) static_win_start_t_ = std::max(t, imu_last_t_);
            ++static_win_count_;
        } else {
            clearStaticWindow_();
        }
    }

    void tryTriggerInitOrZUPT_(double now_t) {
        if (static_win_start_t_ < 0) return;
        const bool time_ok  = (now_t - static_win_start_t_) >= cfg_.zupt_time_min;
        const bool count_ok = static_win_count_ >= cfg_.zupt_min_count;
        if (!(time_ok || count_ok)) return;

        if (!initialized_) {
            doStaticInit_(); initialized_ = true; waiting_init_ = false;
            clearInitBuffer_(); clearStaticWindow_();
        } else {
            applyZUPT_(); nudgeBiasOnStatic_(); clearStaticWindow_();
        }
    }

    void clearInitBuffer_(){ init_acc_buf_.clear(); init_gyr_buf_.clear(); init_t_first_ = -1.0; }
    void clearStaticWindow_(){ static_win_start_t_=-1.0; static_win_count_=0; zupt_acc_buf_.clear(); zupt_gyr_buf_.clear(); }

private:
    Config cfg_;
    State state_;
    ErrorState err_;
    bool initialized_ = false;
    bool waiting_init_ = true;

    // extrinsic cache
    Eigen::Matrix3d R_wi_{Eigen::Matrix3d::Identity()}, R_iw_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d r_wi_in_wheel_{Eigen::Vector3d::Zero()};

    // omega cache (wheel frame)
    Eigen::Vector3d omega_w_last_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d omega_w_prev_{Eigen::Vector3d::Zero()};
    bool omega_w_prev_valid_ = false;

    // init buffers
    double init_t_first_ = -1.0;
    std::vector<Eigen::Vector3d> init_acc_buf_, init_gyr_buf_;

    // static window state
    bool   imu_static_last_   = false; double imu_last_t_   = -1.0;
    bool   wheel_static_last_ = false; double wheel_last_t_ = -1.0; double wheel_last_scaled_ = 0.0;
    double static_win_start_t_ = -1.0; size_t static_win_count_ = 0;
    std::vector<Eigen::Vector3d> zupt_acc_buf_, zupt_gyr_buf_;
};

} // namespace eskf
