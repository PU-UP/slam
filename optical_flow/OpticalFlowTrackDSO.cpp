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
    int half_patch_size = 4;
    int iterations = 10;
    bool inverse = true;

    auto getPixelInfo = [](const Eigen::Vector3f* img, int img_width, int img_height, float x, float y) -> Eigen::Vector3f {
        // 边界检查
        if (x <= 1) x = 2;
        if (y <= 1) y = 2;
        if (x >= img_width - 1) x = img_width - 2;
        if (y >= img_height - 1) y = img_height - 2;

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

    int cost_increase_cnt[6] = {0, 0, 0, 0, 0};
    int nan_cnt[6] = {0, 0, 0, 0, 0};

    std::cout << "keypoints count: " << last_fh_->keypoints.size() << std::endl;

    // iterate all points (todo multi-thread)
    for (auto kp : last_fh_->keypoints) {
        float dx = 0, dy = 0;
        // coarse to fine
        for (int level = start_level; level >= 0; level--) {
            Eigen::Vector2f kp_l = scaleCoordinates(0, level, kp);
            // std::cout << "level: " << level << " kp_l: " << kp_l.transpose() << std::endl;
            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            Eigen::Vector2f b = Eigen::Vector2f::Zero();
            Eigen::Vector2f J;

            float cost = 0, lastCost = 0;
            bool succ = true;

            for (int iter = 0; iter < iterations; iter++) {
                if (!inverse) {
                    H = Eigen::Matrix2f::Zero();
                    b = Eigen::Vector2f::Zero();
                } else {
                    b = Eigen::Vector2f::Zero();
                }

                for (int x = -half_patch_size; x <= half_patch_size; x++) {
                    for (int y = -half_patch_size; y <= half_patch_size; y++) {
                        
                        Eigen::Vector3f last_info = getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], kp_l(0) + x, kp_l(1) + y);
                        Eigen::Vector3f cur_info = getPixelInfo(cur_fh->dIp[level], wG[level], hG[level], kp_l(0) + x + dx, kp_l(1) + y + dy);
                        // error
                        float error = last_info(0) - cur_info(0);
                        cost += error * error;
                        if (!inverse) {
                            J = - 1.0 * Eigen::Vector2f(cur_info[1], cur_info[2]);
                        } else if (iter == 0) {
                            J = - 1.0 * Eigen::Vector2f(last_info[1], last_info[2]);
                            // J = - 1.0 * Eigen::Vector2f(
                            //     0.5 * (getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], kp_l(0) + x + 1, kp_l(1) + y)[0] - 
                            //             getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], kp_l(0) + x - 1, kp_l(1) + y)[0]),
                            //     0.5 * (getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], kp_l(0) + x, kp_l(1) + y + 1)[0] - 
                            //             getPixelInfo(last_fh_->dIp[level], wG[level], hG[level], kp_l(0) + x, kp_l(1) + y - 1)[0])                                
                            // );
                        }

                        b += -error * J;

                        if (!inverse || iter == 0) {
                            H += J * J.transpose();
                        }
                    }
                }

                Eigen::Vector2f update = H.ldlt().solve(b);

                if (std::isnan(update[0])) {
                    // sometimes occurred when we have a black or white patch and H is irreversible
                    std::cout << "update is nan" << std::endl;
                    succ = false;
                    dx = 0;
                    dy = 0;
                    break;
                }

                if (iter > 0 && cost > lastCost) {
                    cost_increase_cnt[level]++;
                    break;
                }

                // update dx, dy
                dx += update[0];
                dy += update[1];
                lastCost = cost;
                succ = true;
                
                if (update.norm() < 1e-2) {
                    // converge
                    break;
                }
            }

            if (succ) {
                if (level > 0) {
                    dx = dx * 2;
                    dy = dy * 2;
                } else {
                    std::cout << "dx, dy: " << dx << "," << dy << std::endl;
                    cur_fh->keypoints.push_back(Eigen::Vector2f(kp(0) + dx, kp(1) + dy));
                }
            } else {
                std::cout << "break at layer: " << level << std::endl; 
                // todo, track failed in coarse layer, should we give it another chance?
                break;
            }
        }
    }

    std::cout << "Tracked count: " << cur_fh->keypoints.size() << std::endl;
    for (int i = start_level; i >= 0; i--) {
        std::cout << "level: " << i << " cost increase cnt: " << cost_increase_cnt[i] << std::endl;
    }

}