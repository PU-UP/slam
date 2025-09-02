#pragma once

// 增量式SFM/VO骨架 + Ceres BA（bundle adjustment）
// - 输入：带里程计位姿的连续图像 RawImageData（T_world_wheel）
// - 通过外参 T_wheel_cam 得到相机先验位姿 T_world_cam
// - ORB + LK 跟踪（简化），用PnP微调；关键帧间三角化建立地标
// - 进行全局BA：优化所有相机位姿与三维点，并对每帧加轮式先验约束
// - 可导出CSV/JSON用于Python可视化
// 依赖：OpenCV >= 4.5, Eigen >= 3.3, Ceres >= 2.0

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <optional>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "data_prepare.hpp"


// SFMOptions is defined in data_prepare.hpp to avoid circular dependency

struct SFMResult {
    std::vector<Eigen::Matrix4d> cam_poses_w_c;
    std::vector<Eigen::Vector3d> points_w;
};

struct TrackObs {
    int frame_idx;
    cv::Point2f px;
};

struct Landmark {
    Eigen::Vector3d Xw;
    std::vector<TrackObs> obs;
    bool is_initialized = false;
};

struct Frame {
    int id = -1;
    double t = 0.0;
    cv::Mat img_gray;
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    std::vector<cv::Point2f> px;
    Eigen::Matrix4d T_w_c = Eigen::Matrix4d::Identity();  // World to Camera transformation (base coordinate system)
};

// ====== Ceres 误差项 ======
struct ReprojError {
    ReprojError(double u, double v, double fx, double fy, double cx, double cy)
        : u_(u), v_(v), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}
    template<typename T>
    bool operator()(const T* const q_xyzw, const T* const t, const T* const Xw, T* residuals) const {
        // q 为世界->相机的四元数（x,y,z,w）; t 为世界->相机平移 twc
        // Xc = Rcw * (Xw - twc)
        T q_cw[4]; // 先从 q_w_c 得到 R_cw = R(q)^T 等价于使用 q 的共轭
        q_cw[0] = -q_xyzw[0];
        q_cw[1] = -q_xyzw[1];
        q_cw[2] = -q_xyzw[2];
        q_cw[3] = q_xyzw[3];
        // 计算 Xc = R(q_cw) * (Xw - t)
        T Xw_minus_t[3];
        Xw_minus_t[0] = Xw[0] - t[0];
        Xw_minus_t[1] = Xw[1] - t[1];
        Xw_minus_t[2] = Xw[2] - t[2];
        T Xc[3];
        ceres::QuaternionRotatePoint(q_cw, Xw_minus_t, Xc);
        // 透视投影
        T xp = Xc[0] / Xc[2];
        T yp = Xc[1] / Xc[2];
        T u = T(fx_) * xp + T(cx_);
        T v = T(fy_) * yp + T(cy_);
        residuals[0] = u - T(u_);
        residuals[1] = v - T(v_);
        return true;
    }
    static ceres::CostFunction* Create(double u, double v, double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojError, 2, 4, 3, 3>(
            new ReprojError(u, v, fx, fy, cx, cy));
    }
    double u_, v_, fx_, fy_, cx_, cy_;
};

struct PosePriorError {
    PosePriorError(const double* q_prior_xyzw, const double* t_prior,
                   double sigma_t, double sigma_r)
        : sigma_t_(sigma_t), sigma_r_(sigma_r) {
        for (int i = 0; i < 4; ++i) q_prior_[i] = q_prior_xyzw[i];
        for (int i = 0; i < 3; ++i) t_prior_[i] = t_prior[i];
    }
    template<typename T>
    bool operator()(const T* const q_xyzw, const T* const t, T* residuals) const {
        // 平移残差
        residuals[0] = (t[0] - T(t_prior_[0])) / T(sigma_t_);
        residuals[1] = (t[1] - T(t_prior_[1])) / T(sigma_t_);
        residuals[2] = (t[2] - T(t_prior_[2])) / T(sigma_t_);
        // 旋转残差：Log( q_prior^{-1} * q )
        T q_prior_inv[4] = { T(-q_prior_[0]), T(-q_prior_[1]), T(-q_prior_[2]), T(q_prior_[3]) };
        T dq[4];
        ceres::QuaternionProduct(q_prior_inv, q_xyzw, dq);
        T aa[3];
        ceres::QuaternionToAngleAxis(dq, aa); // 小角度近似
        residuals[3] = aa[0] / T(sigma_r_);
        residuals[4] = aa[1] / T(sigma_r_);
        residuals[5] = aa[2] / T(sigma_r_);
        return true;
    }
    static ceres::CostFunction* Create(const double* q_prior_xyzw, const double* t_prior,
                                       double sigma_t, double sigma_r) {
        return new ceres::AutoDiffCostFunction<PosePriorError, 6, 4, 3>(
            new PosePriorError(q_prior_xyzw, t_prior, sigma_t, sigma_r));
    }
    double q_prior_[4];
    double t_prior_[3];
    double sigma_t_, sigma_r_;
};

class SFMReconstructor {
public:
    SFMReconstructor(const cv::Mat& cameraK, const cv::Mat& dist, 
                     const Eigen::Matrix4d& T_wheel_cam, const SFMOptions& opts = {});
    
    SFMReconstructor(const CalibrationData& calibration_data, const SFMOptions& opts = {});


    SFMResult Reconstruct(const std::vector<RawImageData>& seq);

    void SetWheelToCam(const Eigen::Matrix4d& T_wheel_cam);

    // 导出CSV/JSON，out_dir 末尾可含/，prefix可为空
    bool SaveForViz(const std::string& out_dir, const std::string& prefix = "") const;

private:
    static cv::Mat toGray(const cv::Mat& img);
    Eigen::Matrix4d wheelPoseToCamPose(const Eigen::Matrix4d& T_w_wheel) const;  // Convert wheel pose (world->wheel) to camera pose (world->camera)

    static void decomposeTcw(const Eigen::Matrix4d& T_w_c, cv::Mat& rvec, cv::Mat& tvec);
    static Eigen::Matrix4d composeTwc(const cv::Mat& rvec, const cv::Mat& tvec);

    void extractFeatures(int i);
    void initTracksFromKeypoints(int);
    void detectAndCompute(int i);
    void extractAndTrack(int i, int j);

    void matchAndAppend(int i, int j);
    
    void refinePosePnP(int i);
    void updateLandmarkObservations(int j);

    void triangulateBetween(int i, int j);

    static bool cheiralityCheck(const Eigen::Matrix4d& T_w_c, const Eigen::Vector3d& Xw);
    bool isGoodBaseline(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B) const;

    cv::Mat projectionFromTwc(const Eigen::Matrix4d& T_w_c) const;
    static cv::Mat RtFromTwc(const Eigen::Matrix4d& T_w_c);
    void pixelToNorm(const std::vector<cv::Point2f>& px, std::vector<cv::Point2f>& norm) const;

    bool projectPoint(const Frame& f, const Eigen::Vector3d& Xw, cv::Point2f& uv) const;
    static int nearestPixel(const std::vector<cv::Point2f>& arr, const cv::Point2f& q, double r);

    // ---------- BA ----------
    static void TwcToQuatTrans(const Eigen::Matrix4d& Twc, double q_xyzw[4], double t[3]);
    static Eigen::Matrix4d QuatTransToTwc(const double q_xyzw[4], const double t[3]);

    void RunBundleAdjustment();
    void reset();

private:
    cv::Mat K_, dist_;
    Eigen::Matrix4d T_wheel_cam_ = Eigen::Matrix4d::Identity();
    SFMOptions opts_;
    cv::Ptr<cv::ORB> orb_;
    std::vector<Frame> frames_;
    std::unordered_map<size_t, Landmark> landmarks_;
    size_t next_landmark_id_ = 0;
};

/*
================= 使用示例 =================

// 1) 构造相机与外参
cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
cv::Mat dist; // 如无畸变可留空
Eigen::Matrix4d T_wheel_cam = ...;  // 由 CalibrationData 组合得到
SFMOptions opts; opts.enable_ba=true; opts.ba_max_iterations=80; opts.prior_trans_sigma=0.05; opts.prior_rot_sigma_rad=2.0*M_PI/180.0;
SFMReconstructor recon(K, dist, T_wheel_cam, opts);

// 2) 重建 + BA
SFMResult result = recon.Reconstruct(query_images);

// 3) 导出可视化文件
recon.SaveForViz("/path/to/out", "run1_");
// 会生成：run1_poses.csv, run1_points.csv, run1_tracks.csv, run1_intrinsics.json

================= CMake 依赖 =================
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE Eigen3::Eigen ceres ${OpenCV_LIBS})

================= 备注 =================
1) 当前track-id与数据关联使用近邻匹配（演示用），工程化请改为稳定的轨迹管理与双向匹配/判定。
2) BA中加入了“轮式先验”，默认把 Reconstruct 流水线得到的 Twc 当作先验。如果你有更精准/独立的轮式估计 Twc_prior，可在 RunBundleAdjustment 中替换为该先验以获得更强约束。
3) 如需优化内参/畸变，可在BA里增加相机内参参数块与对应雅可比（这里保持固定）。
4) 双目/多目扩展：为每个相机建立独立位姿参数块或将多目内参外参与约束一并放入问题。
*/
