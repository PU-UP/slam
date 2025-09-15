#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace eskf {

// ================= 工具 =================
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

// ================ 配置/状态 ================
struct Config {
    // 世界系重力
    Eigen::Vector3d g_world = {0,0,-9.81};

    // 连续时间噪声强度（PSD 的 sqrt）
    double gyro_noise   = 1.5e-3;  // rad/s/sqrt(Hz)
    double accel_noise  = 2.5e-2;  // m/s^2/sqrt(Hz)
    double gyro_rw      = 1.0e-5;  // rad/s^2/sqrt(Hz) (bias random walk)
    double accel_rw     = 1.0e-4;  // m/s^3/sqrt(Hz)

    // 轮速观测
    double wheel_sigma  = 0.1;     // m/s
    double wheel_scale  = 1.0;     // 轮速尺度因子（不优化）

    // 外参
    Eigen::Isometry3d T_bi = Eigen::Isometry3d::Identity(); // body <- imu（仅旋转用于IMU预处理）
    Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity(); // wheel <- body（用于观测与杆臂）

    // —— 静止检测 & ZUPT —— //
    // 判定阈值
    double zupt_speed_th    = 0.05;  // m/s：|scale·wheel| < th
    double zupt_gyro_th     = 0.03;  // rad/s：‖ω_i_raw‖ < th
    double zupt_acc_norm_th = 0.15;  // m/s^2：|‖a_i_raw‖-g| < th
    // 窗口达标条件
    double zupt_time_min    = 1.0;   // s：静止最小时长
    size_t zupt_min_count   = 100;   // 静止最小样本数
    double wheel_stale_max  = 0.5;   // s：轮速“新鲜度”最大容忍

    // ZUPT观测噪声（速度零观测）
    double zupt_sigma_v     = 0.03;  // m/s
    // 静止时对 ba/bg 的“慢调”系数（小）
    double zupt_bias_alpha  = 0.02;
};

struct State {
    double t = 0.0;                              // 当前时间
    Eigen::Vector3d p = Eigen::Vector3d::Zero(); // world
    Eigen::Vector3d v = Eigen::Vector3d::Zero(); // world
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // world<-body

    // 偏置（定义在 IMU系 i）
    Eigen::Vector3d bg_i = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_i = Eigen::Vector3d::Zero();
};

struct ErrorState {
    // 误差态: [δp, δv, δθ, δbg_i, δba_i]
    Eigen::Matrix<double,15,1> x = Eigen::Matrix<double,15,1>::Zero();
    Eigen::Matrix<double,15,15> P = Eigen::Matrix<double,15,15>::Identity() * 1e-3;
};

// ================== ESKF 主类 ==================
class ESKF {
public:
    explicit ESKF(const Config& cfg): cfg_(cfg) {
        // 外参缓存
        R_bi_ = cfg_.T_bi.linear();
        R_ib_ = R_bi_.transpose();

        R_wb_ = cfg_.T_wb.linear();   // wheel <- body
        R_bw_ = R_wb_.transpose();
        r_wb_in_wheel_ = cfg_.T_wb.translation();
        r_bw_in_body_  = R_bw_ * r_wb_in_wheel_;
    }

    // —— 基本接口 —— //
    void setInitial(const State& s0, const ErrorState& e0 = ErrorState()) {
        state_ = s0; err_ = e0;
        initialized_ = true; waiting_init_ = false;
        clearInitBuffer_(); clearStaticWindow_();
    }
    bool   initialized()   const { return initialized_; }
    bool   waitingInit()   const { return waiting_init_; }
    const State&      state() const { return state_; }
    const ErrorState& error() const { return err_;   }

    void   setWheelScale(double s){ cfg_.wheel_scale = s; }
    void   setT_bi(const Eigen::Isometry3d& T_bi){
        cfg_.T_bi = T_bi; R_bi_ = cfg_.T_bi.linear(); R_ib_ = R_bi_.transpose();
    }
    void   setT_wb(const Eigen::Isometry3d& T_wb){
        cfg_.T_wb = T_wb; R_wb_ = cfg_.T_wb.linear(); R_bw_ = R_wb_.transpose();
        r_wb_in_wheel_ = cfg_.T_wb.translation(); r_bw_in_body_ = R_bw_ * r_wb_in_wheel_;
    }

    // —— 预测入口（喂 IMU，原始量在 i 系）—— //
    void predict(double t, const Eigen::Vector3d& w_i_raw, const Eigen::Vector3d& a_i_raw) {
        // 未初始化：仅用于静止检测与初始化缓冲（不推进状态）
        updateImuStatic_(t, w_i_raw, a_i_raw);
        if (!initialized_) {
            tryTriggerInitOrZUPT_(t);
            return;
        }

        // 已初始化：正常预测
        const double dt = std::max(1e-6, t - state_.t);
        state_.t = t;

        // 1) 去偏（i系）
        const Eigen::Vector3d w_i = w_i_raw - state_.bg_i;
        const Eigen::Vector3d a_i = a_i_raw - state_.ba_i;

        // 2) 旋到 b 系
        const Eigen::Vector3d w_b = R_bi_ * w_i;
        const Eigen::Vector3d a_b = R_bi_ * a_i;

        omega_b_last_ = w_b; // 供轮速观测用

        // 3) 推进名义态（world）
        const Eigen::Matrix3d Rwb = state_.q.toRotationMatrix();
        const Eigen::Vector3d a_world = Rwb * a_b + cfg_.g_world;

        state_.v += a_world * dt;
        state_.p += state_.v * dt + 0.5 * a_world * dt * dt;
        state_.q  = RightUpdate(state_.q, w_b * dt);

        // 4) 协方差传播
        propagateCov_(dt, a_b, Rwb);

        // 尝试 ZUPT（若窗口已满足）
        tryTriggerInitOrZUPT_(t);
    }

    // —— 轮速观测入口（标量，带时间戳，推荐用这个）—— //
    // 返回 innovation (预测-观测)，用于调试
    double updateWheel(double t, double wheel_speed_raw) {
        // 更新轮速静止判定
        wheel_last_scaled_ = cfg_.wheel_scale * wheel_speed_raw;
        const bool wheel_static = std::abs(wheel_last_scaled_) < cfg_.zupt_speed_th;
        updateWheelStatic_(t, wheel_static);

        // 尝试 初始化 / ZUPT
        tryTriggerInitOrZUPT_(t);

        if (!initialized_) return 0.0; // 尚未初始化则不做观测更新

        // —— 标量轮速观测更新 —— //
        const Eigen::Matrix3d Rwb = state_.q.toRotationMatrix();
        const Eigen::Matrix3d R_wheel_world = (Rwb * R_bw_).transpose();
        const Eigen::Vector3d v_world_eff = state_.v + Rwb * (omega_b_last_.cross(r_bw_in_body_));
        const double h = (R_wheel_world * v_world_eff).x(); // 预测
        const double z = wheel_last_scaled_;                 // 观测（已缩放）

        // H (1x15)
        Eigen::Matrix<double,1,15> H; H.setZero();
        H.block<1,3>(0,3) = (Eigen::RowVector3d() << 1,0,0).finished() * R_wheel_world;

        // 姿态数值差分
        const double eps = 1e-6;
        for (int k=0;k<3;++k) {
            Eigen::Vector3d d = Eigen::Vector3d::Zero(); d[k]=eps;
            auto f = [&](const Eigen::Quaterniond& q)->double {
                Eigen::Matrix3d Rb = q.toRotationMatrix();
                Eigen::Matrix3d Rww = (Rb * R_bw_).transpose();
                Eigen::Vector3d vwe = state_.v + Rb * (omega_b_last_.cross(r_bw_in_body_));
                return (Rww * vwe).x();
            };
            const double h_plus  = f(RightUpdate(state_.q,  d));
            const double h_minus = f(RightUpdate(state_.q, -d));
            H(0,6+k) = (h_plus - h_minus)/(2.0*eps);
        }

        const double Rm = cfg_.wheel_sigma * cfg_.wheel_sigma;
        const double S  = (H * err_.P * H.transpose())(0,0) + Rm;
        const Eigen::Matrix<double,15,1> K = err_.P * H.transpose() * (1.0/S);

        const double y = z - h;      // 创新
        err_.x += K * y;
        err_.P  = (Eigen::Matrix<double,15,15>::Identity() - K*H) * err_.P;

        injectAndReset_();
        return (h - z);
    }

    // —— 兼容旧接口（不带时间戳；使用当前 state_.t）—— //
    double updateWheel(double wheel_speed_raw) {
        return updateWheel(state_.t, wheel_speed_raw);
    }

    // —— 预留：RTK 位置观测 —— //
    struct RtkMeas {
        double t = 0.0;
        Eigen::Vector3d pos_world = Eigen::Vector3d::Zero();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.25;
        bool has_pos = true;
    };
    void updateRTK(const RtkMeas&){ /* TODO: 实现位置/速度/航向等观测 */ }

private:
    // ========== 协方差传播 ==========
    void propagateCov_(double dt, const Eigen::Vector3d& a_b, const Eigen::Matrix3d& Rwb) {
        Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
        const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

        F.block<3,3>(0,3)  = I3;                    // dp/dv
        F.block<3,3>(3,6)  = - Rwb * Skew(a_b);     // dv/dθ
        F.block<3,3>(3,12) = - Rwb * R_bi_;         // dv/dba_i
        F.block<3,3>(6,9)  = - I3;                  // dθ/dbg_i

        G.block<3,3>(6,0)   = - R_bi_;              // dθ/n_gi
        G.block<3,3>(3,3)   = - Rwb * R_bi_;        // dv/n_ai
        G.block<3,3>(9,6)   =  I3;                  // dbg_i / n_wg_i
        G.block<3,3>(12,9)  =  I3;                  // dba_i / n_wa_i

        const double sg  = cfg_.gyro_noise;
        const double sa  = cfg_.accel_noise;
        const double swg = cfg_.gyro_rw;
        const double swa = cfg_.accel_rw;

        Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
        Qc.block<3,3>(0,0) = (sg*sg)  * Eigen::Matrix3d::Identity();
        Qc.block<3,3>(3,3) = (sa*sa)  * Eigen::Matrix3d::Identity();
        Qc.block<3,3>(6,6) = (swg*swg)* Eigen::Matrix3d::Identity();
        Qc.block<3,3>(9,9) = (swa*swa)* Eigen::Matrix3d::Identity();

        const Eigen::Matrix<double,15,15> Phi = Eigen::Matrix<double,15,15>::Identity() + F*dt;
        const Eigen::Matrix<double,15,15> Qd  = G * Qc * G.transpose() * dt;

        err_.P = Phi * err_.P * Phi.transpose() + Qd;
    }

    // ========== 误差注入 ==========
    void injectAndReset_() {
        state_.p   += err_.x.block<3,1>(0,0);
        state_.v   += err_.x.block<3,1>(3,0);
        state_.q    = RightUpdate(state_.q, err_.x.block<3,1>(6,0));
        state_.bg_i += err_.x.block<3,1>(9,0);
        state_.ba_i += err_.x.block<3,1>(12,0);
        err_.x.setZero();
    }

    // ========== 静止初始化 ==========
    void doStaticInit_() {
        // 用初始化缓冲的均值估计 bg_i/ba_i，并对准重力
        Eigen::Vector3d a_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d w_mean = Eigen::Vector3d::Zero();
        for (auto& a : init_acc_buf_) a_mean += a;
        for (auto& w : init_gyr_buf_) w_mean += w;
        const double N = std::max<size_t>(1, init_acc_buf_.size());
        a_mean /= N; w_mean /= N;

        state_.bg_i = w_mean;

        const Eigen::Vector3d z_w = cfg_.g_world.normalized();
        const Eigen::Vector3d z_i = (-a_mean).normalized();
        const Eigen::Quaterniond q_wi = Eigen::Quaterniond::FromTwoVectors(z_i, z_w);
        const Eigen::Matrix3d R_wi = q_wi.toRotationMatrix();
        const Eigen::Matrix3d R_iw = R_wi.transpose();

        state_.ba_i = a_mean + R_iw * cfg_.g_world; // a ≈ ba_i + R_iw*(-g)

        const Eigen::Matrix3d R_ib = R_bi_.transpose();
        const Eigen::Matrix3d R_wb = R_wi * R_ib;
        state_.q = Eigen::Quaterniond(R_wb).normalized();

        state_.v.setZero();
        // 位置保持默认或由外部设定
        state_.t = init_t_first_ < 0 ? 0.0 : init_t_first_;
    }

    // ========== ZUPT（速度零观测） ==========
    void applyZUPT_() {
        Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
        H.block<3,3>(0,3) = Eigen::Matrix3d::Identity(); // 对速度

        const Eigen::Matrix3d Rm = (cfg_.zupt_sigma_v * cfg_.zupt_sigma_v) * Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d S  = H * err_.P * H.transpose() + Rm;
        const Eigen::Matrix<double,15,3> K = err_.P * H.transpose() * S.inverse();

        const Eigen::Vector3d y = - state_.v; // z - h(x) = 0 - v

        err_.x += K * y;
        err_.P  = (Eigen::Matrix<double,15,15>::Identity() - K*H) * err_.P;
        injectAndReset_();
    }

    // ========== 静止期对 ba/bg 慢调 ==========
    void nudgeBiasOnStatic_() {
        if (zupt_acc_buf_.empty() || zupt_gyr_buf_.empty()) return;

        Eigen::Vector3d a_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d w_mean = Eigen::Vector3d::Zero();
        for (auto& a : zupt_acc_buf_) a_mean += a;
        for (auto& w : zupt_gyr_buf_) w_mean += w;
        a_mean /= double(zupt_acc_buf_.size());
        w_mean /= double(zupt_gyr_buf_.size());

        // 陀螺偏置拉回（i系）
        state_.bg_i = (1.0 - cfg_.zupt_bias_alpha) * state_.bg_i + cfg_.zupt_bias_alpha * w_mean;

        // 加计偏置拉回（i系）： a ≈ ba_i + R_iw*(-g)
        const Eigen::Vector3d z_w = cfg_.g_world.normalized();
        const Eigen::Vector3d z_i = (-a_mean).normalized();
        const Eigen::Quaterniond q_wi = Eigen::Quaterniond::FromTwoVectors(z_i, z_w);
        const Eigen::Matrix3d R_iw = q_wi.toRotationMatrix().transpose();

        const Eigen::Vector3d ba_hat = a_mean + R_iw * cfg_.g_world;
        state_.ba_i = (1.0 - cfg_.zupt_bias_alpha) * state_.ba_i + cfg_.zupt_bias_alpha * ba_hat;
    }

    // ========== IMU静止更新（在 predict() 开头调用） ==========
    void updateImuStatic_(double t, const Eigen::Vector3d& w_i_raw, const Eigen::Vector3d& a_i_raw) {
        // IMU静止判定
        const double g = cfg_.g_world.norm();
        const bool imu_static = (w_i_raw.norm() < cfg_.zupt_gyro_th) &&
                                (std::abs(a_i_raw.norm() - g) < cfg_.zupt_acc_norm_th);
        imu_static_last_ = imu_static;
        imu_last_t_ = t;

        // 静止窗口维护：需要“最近有效”的轮速静止
        const bool wheel_fresh = (wheel_last_t_ >= 0.0) && ((t - wheel_last_t_) <= cfg_.wheel_stale_max);
        const bool both_static = imu_static && wheel_static_last_ && wheel_fresh;
        if (both_static) {
            if (static_win_start_t_ < 0) static_win_start_t_ = std::max(t, wheel_last_t_);
            ++static_win_count_;
        } else {
            clearStaticWindow_();
        }

        // 缓冲（用于首次静止初始化）
        if (imu_static) {
            if (init_t_first_ < 0) init_t_first_ = t;
            init_acc_buf_.push_back(a_i_raw);
            init_gyr_buf_.push_back(w_i_raw);
        }

        // ZUPT期的均值缓冲
        if (both_static) {
            zupt_acc_buf_.push_back(a_i_raw);
            zupt_gyr_buf_.push_back(w_i_raw);
        } else {
            zupt_acc_buf_.clear();
            zupt_gyr_buf_.clear();
        }
    }

    // ========== 轮速静止更新（在 updateWheel() 开头调用） ==========
    void updateWheelStatic_(double t, bool wheel_static) {
        wheel_static_last_ = wheel_static;
        wheel_last_t_ = t;

        // 与最近IMU静止共同判断
        const bool imu_fresh = (imu_last_t_ >= 0.0);
        const bool both_static = wheel_static && imu_static_last_ && imu_fresh && ((t - wheel_last_t_) <= cfg_.wheel_stale_max);
        if (both_static) {
            if (static_win_start_t_ < 0) static_win_start_t_ = std::max(t, imu_last_t_);
            ++static_win_count_;
        } else {
            clearStaticWindow_();
        }
    }

    // ========== 触发：初始化 / ZUPT ==========
    void tryTriggerInitOrZUPT_(double now_t) {
        if (static_win_start_t_ < 0) return;
        const bool time_ok  = (now_t - static_win_start_t_) >= cfg_.zupt_time_min;
        const bool count_ok = static_win_count_ >= cfg_.zupt_min_count;
        if (!(time_ok || count_ok)) return;

        if (!initialized_) {
            doStaticInit_();
            initialized_ = true; waiting_init_ = false;
            clearInitBuffer_(); clearStaticWindow_();
        } else {
            applyZUPT_();
            nudgeBiasOnStatic_();
            clearStaticWindow_();
        }
    }

    // ========== 清理工具 ==========
    void clearInitBuffer_(){ init_acc_buf_.clear(); init_gyr_buf_.clear(); init_t_first_ = -1.0; }
    void clearStaticWindow_(){
        static_win_start_t_ = -1.0; static_win_count_ = 0;
        zupt_acc_buf_.clear(); zupt_gyr_buf_.clear();
    }

private:
    Config cfg_;
    State state_;
    ErrorState err_;
    bool initialized_  = false;
    bool waiting_init_ = true;

    // 外参缓存
    Eigen::Matrix3d R_bi_{Eigen::Matrix3d::Identity()}, R_ib_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d R_wb_{Eigen::Matrix3d::Identity()}, R_bw_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d r_wb_in_wheel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d r_bw_in_body_{Eigen::Vector3d::Zero()};

    // 最近一次（b系）角速度（供轮速观测）
    Eigen::Vector3d omega_b_last_{Eigen::Vector3d::Zero()};

    // —— 初始化缓冲 —— 
    double init_t_first_ = -1.0;
    std::vector<Eigen::Vector3d> init_acc_buf_, init_gyr_buf_;

    // —— ZUPT 窗口状态 —— 
    bool   imu_static_last_   = false;
    double imu_last_t_        = -1.0;

    bool   wheel_static_last_ = false;
    double wheel_last_t_      = -1.0;
    double wheel_last_scaled_ = 0.0;

    double static_win_start_t_ = -1.0;
    size_t static_win_count_   = 0;

    std::vector<Eigen::Vector3d> zupt_acc_buf_, zupt_gyr_buf_;
};

} // namespace eskf
