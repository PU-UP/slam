#include "sfm_reconstructor.hpp"

// ====== SFMReconstructor 实现 ======

SFMReconstructor::SFMReconstructor(const cv::Mat& cameraK, const cv::Mat& dist, 
                                   const Eigen::Matrix4d& T_wheel_cam, const SFMOptions& opts)
    : K_(cameraK.clone()), dist_(dist.clone()), T_wheel_cam_(T_wheel_cam), opts_(opts) {
    CV_Assert(K_.rows == 3 && K_.cols == 3);
    orb_ = cv::ORB::create(opts_.max_features);
}

SFMReconstructor::SFMReconstructor(const CalibrationData& calibration_data, const SFMOptions& opts)
    : opts_(opts) {
    // Extract camera matrix from calibration data
    const auto& proj_params = calibration_data.intrinsic_camera.projection_parameters;
    const auto& dist_params = calibration_data.intrinsic_camera.distortion_parameters;
    
    K_ = (cv::Mat_<double>(3, 3) << 
          proj_params[0], 0, proj_params[2],
          0, proj_params[1], proj_params[3],
          0, 0, 1);
    
    // Set distortion coefficients - use 5-element format for OpenCV (k1, k2, p1, p2, k3)
    // If all distortion params are zero, use empty distortion matrix
    if (dist_params[0] == 0 && dist_params[1] == 0 && dist_params[2] == 0 && dist_params[3] == 0) {
        dist_ = cv::Mat::zeros(1, 5, CV_64F);
    } else {
        dist_ = (cv::Mat_<double>(1, 5) << 
                dist_params[0], dist_params[1], 0, 0, dist_params[2]);
    }
    
    // Set wheel to camera transformation
    T_wheel_cam_ = calibration_data.extrinsic_wheel_T_cam0.transform;
    
    orb_ = cv::ORB::create(opts_.max_features);
}

SFMResult SFMReconstructor::Reconstruct(const std::vector<RawImageData>& seq) {
    reset();
    if (seq.empty()) return {};
    
    frames_.reserve(seq.size());
    for (const auto& r : seq) {
        Frame f;
        f.id = r.id;
        f.t = r.timestamp;
        f.img_gray = toGray(r.image);
        f.T_w_c = wheelPoseToCamPose(r.wheel_pose);
        frames_.push_back(std::move(f));
    }
    
    extractFeatures(0);
    initTracksFromKeypoints(0);
    
    int last_kf = 0;
    for (int i = 1; i < (int)frames_.size(); ++i) {
        extractAndTrack(i - 1, i);
        if ((int)frames_[i].px.size() < opts_.min_tracked_for_pnp) {
            detectAndCompute(i);
            matchAndAppend(i - 1, i);
        }
        if (opts_.refine_with_pnp) refinePosePnP(i);
        updateLandmarkObservations(i);        // 将已有码点尽可能关联为观测
        triangulateBetween(last_kf, i);
        if (isGoodBaseline(frames_[last_kf].T_w_c, frames_[i].T_w_c)) {
            last_kf = i;
        }
    }
    
    if (opts_.enable_ba) RunBundleAdjustment();
    
    SFMResult res;
    res.cam_poses_w_c.reserve(frames_.size());
    for (auto& f : frames_) {
        res.cam_poses_w_c.push_back(f.T_w_c);
    }
    
    res.points_w.reserve(landmarks_.size());
    for (auto& kv : landmarks_) {
        if (kv.second.is_initialized) {
            res.points_w.push_back(kv.second.Xw);
        }
    }
    return res;
}

void SFMReconstructor::SetWheelToCam(const Eigen::Matrix4d& T_wheel_cam) { 
    T_wheel_cam_ = T_wheel_cam; 
}

bool SFMReconstructor::SaveForViz(const std::string& out_dir, const std::string& prefix) const {
    std::string pp = out_dir;
    if (!pp.empty() && pp.back() != '/') pp.push_back('/');
    
    std::string poses_fn = pp + prefix + "poses.csv";
    std::string points_fn = pp + prefix + "points.csv";
    std::string tracks_fn = pp + prefix + "tracks.csv";
    std::string K_fn = pp + prefix + "intrinsics.json";
    
    std::ofstream fp(poses_fn), fx(points_fn), ft(tracks_fn), fk(K_fn);
    if (!fp || !fx || !ft || !fk) {
        std::cerr << "[SaveForViz] open file failed" << std::endl;
        return false;
    }
    
    // poses.csv: frame_idx,id,timestamp,tx,ty,tz,qx,qy,qz,qw
    fp << "frame_idx,id,timestamp,tx,ty,tz,qx,qy,qz,qw\n";
    for (int i = 0; i < (int)frames_.size(); ++i) {
        const auto& f = frames_[i];
        Eigen::Quaterniond q(f.T_w_c.block<3, 3>(0, 0));
        Eigen::Vector3d t = f.T_w_c.block<3, 1>(0, 3);
        fp << i << "," << f.id << "," << std::fixed << f.t << ","
           << t.x() << "," << t.y() << "," << t.z() << ","
           << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
    }
    
    // points.csv: lm_id,X,Y,Z
    fx << "landmark_id,X,Y,Z\n";
    for (auto& kv : landmarks_) {
        if (kv.second.is_initialized) {
            auto id = kv.first;
            auto& X = kv.second.Xw;
            fx << id << "," << X.x() << "," << X.y() << "," << X.z() << "\n";
        }
    }
    
    // tracks.csv: lm_id,frame_idx,u,v
    ft << "landmark_id,frame_idx,u,v\n";
    for (auto& kv : landmarks_) {
        if (!kv.second.is_initialized) continue;
        for (auto& ob : kv.second.obs) {
            ft << kv.first << "," << ob.frame_idx << "," 
               << ob.px.x << "," << ob.px.y << "\n";
        }
    }
    
    // intrinsics.json
    double fxv = K_.at<double>(0, 0), fyv = K_.at<double>(1, 1);
    double cxv = K_.at<double>(0, 2), cyv = K_.at<double>(1, 2);
    int w = 0, h = 0;
    if (!frames_.empty()) {
        w = frames_[0].img_gray.cols;
        h = frames_[0].img_gray.rows;
    }
    fk << "{\n  \"fx\": " << fxv << ",\n  \"fy\": " << fyv 
       << ",\n  \"cx\": " << cxv << ",\n  \"cy\": " << cyv
       << ",\n  \"width\": " << w << ",\n  \"height\": " << h << "\n}\n";
    return true;
}

// ====== 私有方法实现 ======

cv::Mat SFMReconstructor::toGray(const cv::Mat& img) {
    if (img.empty()) return img;
    if (img.channels() == 1) return img.clone();
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

Eigen::Matrix4d SFMReconstructor::wheelPoseToCamPose(const Eigen::Matrix4d& T_w_wheel) const {
    return T_w_wheel * T_wheel_cam_;
}

void SFMReconstructor::decomposeTcw(const Eigen::Matrix4d& T_w_c, cv::Mat& rvec, cv::Mat& tvec) {
    Eigen::Matrix3d Rwc = T_w_c.block<3, 3>(0, 0);
    Eigen::Vector3d twc = T_w_c.block<3, 1>(0, 3);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;
    cv::Mat Rcv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Rcv.at<double>(r, c) = Rcw(r, c);
        }
    }
    cv::Rodrigues(Rcv, rvec);
    tvec = (cv::Mat_<double>(3, 1) << tcw.x(), tcw.y(), tcw.z());
}

Eigen::Matrix4d SFMReconstructor::composeTwc(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R_cw;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_cw(r, c) = Rcv.at<double>(r, c);
        }
    }
    Eigen::Vector3d t_cw(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    Eigen::Matrix3d R_wc = R_cw.transpose();
    Eigen::Vector3d t_wc = -R_wc * t_cw;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R_wc;
    T.block<3, 1>(0, 3) = t_wc;
    return T;
}

void SFMReconstructor::extractFeatures(int i) {
    auto& f = frames_[i];
    orb_->detectAndCompute(f.img_gray, cv::noArray(), f.kps, f.desc);
    f.px.clear();
    f.px.reserve(f.kps.size());
    for (auto& kp : f.kps) {
        f.px.emplace_back(kp.pt);
    }
}

void SFMReconstructor::initTracksFromKeypoints(int) { 
    /* 简化：不显式维护track-id */ 
}

void SFMReconstructor::detectAndCompute(int i) {
    auto& f = frames_[i];
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    orb_->detectAndCompute(f.img_gray, cv::noArray(), kps, desc);
    if (!desc.empty()) {
        if (f.desc.empty()) {
            f.desc = desc.clone();
        } else {
            cv::vconcat(f.desc, desc, f.desc);
        }
        f.kps.insert(f.kps.end(), kps.begin(), kps.end());
        for (auto& kp : kps) {
            f.px.push_back(kp.pt);
        }
    }
}

void SFMReconstructor::extractAndTrack(int i, int j) {
    auto& fi = frames_[i];
    auto& fj = frames_[j];
    if (opts_.use_optical_flow_tracking && !fi.px.empty()) {
        std::vector<cv::Point2f> nextPts;
        std::vector<unsigned char> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(fi.img_gray, fj.img_gray, fi.px, nextPts, status, err,
            cv::Size(opts_.klt_win_size, opts_.klt_win_size), opts_.klt_max_level,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                           opts_.klt_max_iter, opts_.klt_eps));
        fj.px.clear();
        fj.px.reserve(nextPts.size());
        for (size_t k = 0; k < nextPts.size(); ++k) {
            if (status[k]) fj.px.push_back(nextPts[k]);
        }
    } else {
        extractFeatures(j);
    }
}

void SFMReconstructor::matchAndAppend(int i, int j) {
    auto& fi = frames_[i];
    auto& fj = frames_[j];
    if (fi.desc.empty() || fj.desc.empty()) return;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(fi.desc, fj.desc, knn, 2);

    fj.px.clear(); // 保证是匹配得到的点
    std::vector<cv::Point2f> pts_i, pts_j;

    for (auto& v : knn) {
        if (v.size() >= 2 && v[0].distance < opts_.match_ratio * v[1].distance) {
            const cv::KeyPoint& kpi = fi.kps[v[0].queryIdx];
            const cv::KeyPoint& kpj = fj.kps[v[0].trainIdx];
            pts_i.push_back(kpi.pt);
            pts_j.push_back(kpj.pt);
        }
    }

    // 这里只把 j 的点保存到 fj.px，方便后续
    fj.px = pts_j;

    // 如果你需要保留两帧的对应关系，可以额外存储匹配对
    // matches_ij.push_back({pts_i[k], pts_j[k]});
}

void SFMReconstructor::refinePosePnP(int i) {
    auto& f = frames_[i];
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    obj.reserve(landmarks_.size());
    img.reserve(landmarks_.size());
    
    const int max_use = 500;
    int used = 0;
    for (const auto& kv : landmarks_) {
        if (!kv.second.is_initialized) continue;
        cv::Point2f uv;
        if (!projectPoint(f, kv.second.Xw, uv)) continue;
        int idx = nearestPixel(f.px, uv, 3.0);
        if (idx >= 0) {
            obj.emplace_back((float)kv.second.Xw.x(), (float)kv.second.Xw.y(), (float)kv.second.Xw.z());
            img.emplace_back(f.px[idx]);
            if (++used >= max_use) break;
        }
    }
    
    if ((int)obj.size() < opts_.min_tracked_for_pnp) return;
    
    cv::Mat rvec, tvec;
    decomposeTcw(f.T_w_c, rvec, tvec);
    bool use_guess = true;
    cv::Mat inliers;
    bool ok = cv::solvePnPRansac(obj, img, K_, dist_, rvec, tvec, use_guess, 100,
                                opts_.ransac_reproj_thresh, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    if (ok) f.T_w_c = composeTwc(rvec, tvec);
}

void SFMReconstructor::updateLandmarkObservations(int j) {
    auto& fj = frames_[j];
    for (auto& kv : landmarks_) {
        if (!kv.second.is_initialized) continue;
        cv::Point2f uv;
        if (!projectPoint(fj, kv.second.Xw, uv)) continue;
        int idx = nearestPixel(fj.px, uv, 2.5);
        if (idx >= 0) {
            kv.second.obs.push_back({j, fj.px[idx]});
        }
    }
}

void SFMReconstructor::triangulateBetween(int i, int j) {
    auto& fi = frames_[i];
    auto& fj = frames_[j];
    if (fi.px.empty() || fj.px.empty()) return;
    
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& p2 : fj.px) {
        int idx = nearestPixel(fi.px, p2, 2.5);
        if (idx >= 0) {
            pts1.push_back(fi.px[idx]);
            pts2.push_back(p2);
        }
        if (pts1.size() >= 1000) break;
    }
    if (pts1.size() < 12) return;
    
    std::vector<cv::Point2f> n1 = pts1, n2 = pts2;
    if (opts_.undistort && !dist_.empty()) {
        cv::undistortPoints(pts1, n1, K_, dist_);
        cv::undistortPoints(pts2, n2, K_, dist_);
    } else {
        pixelToNorm(pts1, n1);
        pixelToNorm(pts2, n2);
    }
    
    cv::Mat P1 = RtFromTwc(fi.T_w_c), P2 = RtFromTwc(fj.T_w_c);
    cv::Mat X4;
    cv::triangulatePoints(P1, P2, n1, n2, X4);
    
    for (int c = 0; c < X4.cols; ++c) {
        cv::Mat x = X4.col(c);
        double w = x.at<float>(3);
        if (std::abs(w) < 1e-6) continue;
        
        Eigen::Vector4d Xh(x.at<float>(0) / w, x.at<float>(1) / w, x.at<float>(2) / w, 1.0);
        Eigen::Vector3d Xw = Xh.head<3>();
        if (!cheiralityCheck(fi.T_w_c, Xw)) continue;
        if (!cheiralityCheck(fj.T_w_c, Xw)) continue;
        
        Landmark lm;
        lm.Xw = Xw;
        lm.is_initialized = true;
        lm.obs.push_back({i, pts1[c]});
        lm.obs.push_back({j, pts2[c]});
        landmarks_.emplace(next_landmark_id_++, std::move(lm));
    }
}

bool SFMReconstructor::cheiralityCheck(const Eigen::Matrix4d& T_w_c, const Eigen::Vector3d& Xw) {
    Eigen::Matrix3d Rwc = T_w_c.block<3, 3>(0, 0);
    Eigen::Vector3d twc = T_w_c.block<3, 1>(0, 3);
    Eigen::Vector3d Xc = Rwc.transpose() * (Xw - twc);
    return Xc.z() > 0;
}

bool SFMReconstructor::isGoodBaseline(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B) const {
    Eigen::Vector3d ta = A.block<3, 1>(0, 3), tb = B.block<3, 1>(0, 3);
    double dt = (ta - tb).norm();
    if (dt > 0.10) return true;
    
    Eigen::Matrix3d Ra = A.block<3, 3>(0, 0), Rb = B.block<3, 3>(0, 0);
    double angle = std::acos(std::min(1.0, std::max(-1.0, 
        ((Ra.transpose() * Rb).trace() - 1) / 2.0)));
    return angle * 180.0 / M_PI > opts_.triang_min_parallax_deg;
}

cv::Mat SFMReconstructor::projectionFromTwc(const Eigen::Matrix4d& T_w_c) const {
    cv::Mat Rt = RtFromTwc(T_w_c);
    return K_ * Rt;
}

cv::Mat SFMReconstructor::RtFromTwc(const Eigen::Matrix4d& T_w_c) {
    Eigen::Matrix3d Rwc = T_w_c.block<3, 3>(0, 0);
    Eigen::Vector3d twc = T_w_c.block<3, 1>(0, 3);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;
    cv::Mat Rt(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Rt.at<double>(r, c) = Rcw(r, c);
        }
    }
    Rt.at<double>(0, 3) = tcw.x();
    Rt.at<double>(1, 3) = tcw.y();
    Rt.at<double>(2, 3) = tcw.z();
    return Rt;
}

void SFMReconstructor::pixelToNorm(const std::vector<cv::Point2f>& px, std::vector<cv::Point2f>& norm) const {
    norm.resize(px.size());
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);
    for (size_t i = 0; i < px.size(); ++i) {
        norm[i].x = (px[i].x - (float)cx) / (float)fx;
        norm[i].y = (px[i].y - (float)cy) / (float)fy;
    }
}

bool SFMReconstructor::projectPoint(const Frame& f, const Eigen::Vector3d& Xw, cv::Point2f& uv) const {
    cv::Mat Rt = RtFromTwc(f.T_w_c);
    cv::Mat X(3, 1, CV_64F);
    X.at<double>(0) = Xw.x();
    X.at<double>(1) = Xw.y();
    X.at<double>(2) = Xw.z();
    cv::Mat x = Rt(cv::Rect(0, 0, 3, 3)) * X + Rt.col(3);
    double Z = x.at<double>(2);
    if (Z <= 0) return false;
    
    double xn = x.at<double>(0) / Z, yn = x.at<double>(1) / Z;
    // Always use simple projection without distortion to avoid cv::projectPoints issues
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);
    uv.x = (float)(fx * xn + cx);
    uv.y = (float)(fy * yn + cy);
    return true;
}

int SFMReconstructor::nearestPixel(const std::vector<cv::Point2f>& arr, const cv::Point2f& q, double r) {
    if (arr.empty()) return -1;
    int best = -1;
    double bestd = r * r;
    for (int i = 0; i < (int)arr.size(); ++i) {
        double dx = arr[i].x - q.x, dy = arr[i].y - q.y;
        double d = dx * dx + dy * dy;
        if (d < bestd) {
            bestd = d;
            best = i;
        }
    }
    return best;
}

void SFMReconstructor::TwcToQuatTrans(const Eigen::Matrix4d& Twc, double q_xyzw[4], double t[3]) {
    Eigen::Quaterniond q(Twc.block<3, 3>(0, 0));
    q.normalize();
    q_xyzw[0] = q.x();
    q_xyzw[1] = q.y();
    q_xyzw[2] = q.z();
    q_xyzw[3] = q.w();
    Eigen::Vector3d tt = Twc.block<3, 1>(0, 3);
    t[0] = tt.x();
    t[1] = tt.y();
    t[2] = tt.z();
}

Eigen::Matrix4d SFMReconstructor::QuatTransToTwc(const double q_xyzw[4], const double t[3]) {
    Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
    q.normalize();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = t[0];
    T(1, 3) = t[1];
    T(2, 3) = t[2];
    return T;
}

void SFMReconstructor::RunBundleAdjustment() {
    // 准备参数块
    const int N = (int)frames_.size();
    std::vector<std::array<double, 7>> pose_params(N); // [qx qy qz qw | tx ty tz]
    for (int i = 0; i < N; ++i) {
        double q[4], t[3];
        TwcToQuatTrans(frames_[i].T_w_c, q, t);
        pose_params[i] = {q[0], q[1], q[2], q[3], t[0], t[1], t[2]};
    }

    std::vector<size_t> lm_ids;
    lm_ids.reserve(landmarks_.size());
    for (auto& kv : landmarks_) {
        if (kv.second.is_initialized) lm_ids.push_back(kv.first);
    }
    const int M = (int)lm_ids.size();
    std::vector<std::array<double, 3>> point_params(M);
    for (int m = 0; m < M; ++m) {
        auto& X = landmarks_[lm_ids[m]].Xw;
        point_params[m] = {X.x(), X.y(), X.z()};
    }

    ceres::Problem problem;
    auto* quat_param = new ceres::QuaternionParameterization();
    
    // 添加位姿参数块
    for (int i = 0; i < N; ++i) {
        problem.AddParameterBlock(pose_params[i].data() + 0, 4, quat_param);
        problem.AddParameterBlock(pose_params[i].data() + 4, 3);
        if (opts_.fix_first_pose && i == 0) {
            problem.SetParameterBlockConstant(pose_params[i].data() + 0);
            problem.SetParameterBlockConstant(pose_params[i].data() + 4);
        }
    }
    
    // 相机内参
    const double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    const double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);

    // 观测误差
    ceres::LossFunction* loss = new ceres::HuberLoss(opts_.ba_huber_delta_px);
    for (int m = 0; m < M; ++m) {
        auto& lm = landmarks_[lm_ids[m]];
        for (auto& ob : lm.obs) {
            int i = ob.frame_idx;
            ceres::CostFunction* cost = ReprojError::Create(ob.px.x, ob.px.y, fx, fy, cx, cy);
            problem.AddResidualBlock(cost, loss, 
                pose_params[i].data() + 0, pose_params[i].data() + 4, point_params[m].data());
        }
    }
    
    // 轮式先验（每帧）
    for (int i = 0; i < N; ++i) {
        double q_prior[4], t_prior[3];
        // 这里使用当前估计做初值 & 作为先验：若你有单独的轮式先验Twc_prior，请改为那个
        TwcToQuatTrans(frames_[i].T_w_c, q_prior, t_prior);
        ceres::CostFunction* prior = PosePriorError::Create(q_prior, t_prior, 
            opts_.prior_trans_sigma, opts_.prior_rot_sigma_rad);
        problem.AddResidualBlock(prior, nullptr, 
            pose_params[i].data() + 0, pose_params[i].data() + 4);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = opts_.ba_max_iterations;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;

    // 写回结果
    for (int i = 0; i < N; ++i) {
        frames_[i].T_w_c = QuatTransToTwc(pose_params[i].data() + 0, pose_params[i].data() + 4);
    }
    for (int m = 0; m < M; ++m) {
        landmarks_[lm_ids[m]].Xw = Eigen::Vector3d(point_params[m][0], point_params[m][1], point_params[m][2]);
    }
}

void SFMReconstructor::reset() {
    frames_.clear();
    landmarks_.clear();
    next_landmark_id_ = 0;
}