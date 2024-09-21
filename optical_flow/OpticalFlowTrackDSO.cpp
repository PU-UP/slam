#include "OpticalFLowTrackerDSO.hpp"

void OpticalFlowTrackerDSO::track(FrameHessian* cur_fh) {
    // set first to track
    if (last_fh_ == nullptr) {
        last_fh_ = cur_fh;
        std::cout << "to tracked image is set" << std::endl;
        return;
    }

    // some parameters
    int start_level = pyrLevelsUsed - 1;
    int half_patch_size = 10;
    int iterations = 10;
    bool inverse = true;

    auto getPixelInfo = [](const Eigen::Vector3f* img, int img_width, int img_height, float x, float y) -> Eigen::Vector3f {
        // 边界检查
        if (x <= 1) x = 1;
        if (y <= 1) y = 1;
        if (x >= img_width - 1) x = img_width - 1;
        if (y >= img_height - 1) y = img_height - 1;

        // 双线性插值系数
        float xx = x - std::floor(x);
        float yy = y - std::floor(y);

        // 四个相邻像素的位置
        int x_int = std::floor(x);
        int y_int = std::floor(y);
        int x_a1 = std::min(img_width - 1, x_int + 1);
        int y_a1 = std::min(img_height - 1, y_int + 1);

        // 计算插值
        Eigen::Vector3f v00 = img[y_int * img_width + x_int];     // 左上角
        Eigen::Vector3f v01 = img[y_int * img_width + x_a1];      // 右上角
        Eigen::Vector3f v10 = img[y_a1 * img_width + x_int];      // 左下角
        Eigen::Vector3f v11 = img[y_a1 * img_width + x_a1];       // 右下角

        Eigen::Vector3f interpolated_value = (1 - xx) * (1 - yy) * v00   // 左上角
                                        + xx * (1 - yy) * v01          // 右上角
                                        + (1 - xx) * yy * v10          // 左下角
                                        + xx * yy * v11;               // 右下角

        return interpolated_value;
        
    };

    auto scaleCoordinates = [](int current_level, int target_level, const Eigen::Vector2f& pt) -> Eigen::Vector2f {
        int level_diff = current_level - target_level;  // 计算层级差异
        float scaleFactor = std::pow(2, level_diff);    // 根据层级差异计算缩放因子
        return pt * scaleFactor;  // 返回缩放后的坐标点
    };

    int cost_increase_cnt[6] = {0, 0, 0, 0, 0, 0};
    int occilation_cnt[6] = {0, 0, 0, 0, 0, 0};
    int conv_cnt[6] = {0, 0, 0, 0, 0, 0};
    int out_loop_cnt[6] = {0, 0, 0, 0, 0, 0};
    int badH[6] = {0, 0, 0, 0, 0, 0};
    int outbound[6] = {0, 0, 0, 0, 0, 0};

    std::cout << "keypoints count: " << last_fh_->keypoints.size() << std::endl;
    int tracked_cnt = 0;

    // border margin
    int bm = 3;

    static int pt_id = 0;

    // iterate all points (todo multi-thread)
    for (auto kp : last_fh_->keypoints) {
        pt_id++;
        float dx = 0, dy = 0;
        // coarse to fine
        for (int level = start_level; level >= 0; level--) {
            Eigen::Vector2f kp_l = scaleCoordinates(0, level, kp);

            // std::cout << "level: " << level << " kp_l: " << kp_l.transpose() << std::endl;
            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            Eigen::Vector2f b = Eigen::Vector2f::Zero();
            Eigen::Vector2f J;

            float cost = 0, lastCost = 0;
            bool succ = false;
            Eigen::Vector2f last_update = Eigen::Vector2f::Zero();

            for (int iter = 0; iter < iterations; iter++) {
                if (!inverse) {
                    H = Eigen::Matrix2f::Zero();
                    b = Eigen::Vector2f::Zero();
                } else {
                    b = Eigen::Vector2f::Zero();
                }
                int used_cnt = 0;
                for (int x = -half_patch_size; x <= half_patch_size; x++) {
                    for (int y = -half_patch_size; y <= half_patch_size; y++) {
                        if (kp_l(0) + x <= bm || kp_l(0) + x + dx <= bm ||
                            kp_l(0) + x >= wG[level] - bm || kp_l(0) + x + dx >= wG[level] - bm ||
                            kp_l(1) + y <= bm || kp_l(1) + y + dy <= bm ||
                            kp_l(1) + y >= hG[level] - bm || kp_l(1) + y + dy >= hG[level] - bm
                        ) {
                            continue;
                        }
                        used_cnt++;
                        Eigen::Vector3f last_info = getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], 
                                                                    kp_l(0) + x, kp_l(1) + y);
                        Eigen::Vector3f cur_info = getPixelInfo(cur_fh->dIp[level], wG[level], hG[level], 
                                                                    kp_l(0) + x + dx, kp_l(1) + y + dy);
                        // error
                        float error = last_info(0) - cur_info(0);
                        // std::cout << "error: " << error << "\tlast: " << last_info(0) << "\tcur: " << cur_info(0) << std::endl;
                        cost += error * error;
                        if (!inverse) {
                            J = - 1.0 * Eigen::Vector2f(cur_info[1], cur_info[2]);
                        } else if (iter == 0) {
                            J = - 1.0 * Eigen::Vector2f(last_info[1], last_info[2]);
                        }

                        b += -error * J;

                        if (!inverse || iter == 0) {
                            H += J * J.transpose();
                        }
                    }
                }


                if (used_cnt < 1) {
                    succ = false;
                    break;
                }

                float A22 = H(1, 1);
                float A11 = H(0, 0);
                float A12 = H(0, 1);
                float D = A11*A22 - A12*A12;
                float winSize = half_patch_size * 2 + 1;
                float minEig = (A22 + A11 - sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2 * winSize * winSize);
                // float minEig = (A22 + A11 - sqrt((A11-A22)*(A11-A22) +
                //         4.f*A12*A12))/(2 * used_cnt);
                if(minEig < 1e-4 || D < 1e-4 ) {
                    badH[level]++;
                    std::cout << "[badH] minEig: " << minEig << " D: " << D << std::endl;
                    break;
                }

                // Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver(H);
                // if (eigensolver.info() != Eigen::Success) {
                //     // Handle the error if eigenvalue computation fails.
                //     std::cout << "Eigenvalue decomposition failed" << std::endl;
                //     return;
                // }

                // Eigen::Vector2f eigenvalues = eigensolver.eigenvalues();

                // // Minimum eigenvalue check
                // float min_eigenvalue = eigenvalues.minCoeff();
                // float min_eig_threshold = 1e-4;  // Set this threshold based on your needs

                // if (min_eigenvalue < min_eig_threshold) {
                //     badH[level]++;
                //     std::cout << "[badH] pt id: " << pt_id << "\t lvl: " << level << " iter: " << iter
                //               << " minEig: " << min_eigenvalue << "\tcost: " << cost << " used cnt: " << used_cnt
                //               << std::endl;
                //     succ = false;
                //     break;
                // }

                Eigen::Vector2f update = H.ldlt().solve(b);

                if (std::isnan(update[0])) {
                    // sometimes occurred when we have a black or white patch and H is irreversible
                    std::cout << "update is nan" << std::endl;
                    succ = false;
                    dx = 0;
                    dy = 0;
                    break;
                }

                if (kp_l(0) + dx + update(0) < 0 || kp_l(1) + dy + update(1) > wG[level]) {
                    succ = false;
                    outbound[level]++;
                    break;
                }

                if (iter > 0 && cost > lastCost) {
                    cost_increase_cnt[level]++;
                    std::cout << "[INC] pt id: " << pt_id << "\t lvl: " << level << " iter: " << iter
                              << " update: " << update.transpose() << "\tcost: " << cost
                              << std::endl;
                    // succ = true;
                    // break;
                }

                if (iter > 0 && (abs(last_update(0) + update(0)) < 1e-2) && 
                                (abs(last_update(1) + update(1)) < 1e-2) && level > 0) {
                    std::cout << "[OCC] pt id: " << pt_id << "\t lvl: " << level << " iter: " << iter
                              << " last update: " << last_update.transpose() 
                              << "\tcurr update: " << update.transpose() 
                              << "\tcost: " << cost
                              << std::endl;
                    dx += 0.5 * update[0];
                    dy += 0.5 * update[1];
                    occilation_cnt[level]++;
                    succ = true;
                    break; // go to next level with half update
                }

                // update dx, dy
                dx += update[0];
                dy += update[1];

                last_update = update;
                
                // if (update.norm() < 1e-2) {
                if (update.dot(update) < 1e-2) {
                    conv_cnt[level]++;
                    std::cout << "[COV] pt id: " << pt_id << "\t lvl: " << level << " iter: " << iter
                              << " update: " << update.transpose() << "\tcost: " << cost
                              << std::endl;
                    succ = true;
                    // converge
                    break;
                }

                if (iter == iterations - 1) {
                    out_loop_cnt[level]++;
                    std::cout << "[OUT] pt id: " << pt_id << "\t lvl: " << level << " iter: " << iter
                              << " update: " << update.transpose() << "\tcost: " << cost
                              << std::endl;
                    succ = false;
                    break;
                }
                lastCost = cost;
                cost = 0;
                succ = true;
            }

            if (succ) {
                if (level > 0) {
                    dx = dx * 2;
                    dy = dy * 2;
                } else {
                    // std::cout << "dx, dy: " << dx << "," << dy << std::endl;
                    tracked_cnt++;
                    cur_fh->keypoints.push_back(Eigen::Vector2f(kp(0) + dx, kp(1) + dy));
                }
            } else {
                // std::cout << "break at level: " << level << std::endl; 
                // todo, track failed in coarse layer, should we give it another chance?
                if (level == 0) {
                    cur_fh->keypoints.push_back(Eigen::Vector2f::Zero());
                    break;
                } else {
                    dx = dx * 2;
                    dy = dy * 2;
                }
            }
        }
    }

    std::cout << "Tracked count: " << tracked_cnt << " out of " << cur_fh->keypoints.size() << std::endl;
    for (int i = start_level; i >= 0; i--) {
        std::cout << "level: " << i << " conv cnt: " << conv_cnt[i];
        // std::cout << "level: " << i << " cost inc: " << cost_increase_cnt[i] << std::endl;
        std::cout << "\tout  cnt: " << out_loop_cnt[i];
        std::cout << "\tocci cnt: " << occilation_cnt[i];
        std::cout << "\tbadH cnt: " << badH[i];
        std::cout << "\toutbound cnt: " << outbound[i];
        std::cout << std::endl;
    }

}