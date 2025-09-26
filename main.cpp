#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include "data_prepare.hpp"
#include "include/slam/modules.hpp"

#include "eskf/eskf.hpp"

// 全局控制变量
std::atomic<bool> is_paused(false);
std::atomic<bool> should_exit(false);

// 键盘输入处理函数
void keyboardInputHandler() {
    // 设置终端为非阻塞模式
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    
    char c;
    while (!should_exit.load()) {
        if (read(STDIN_FILENO, &c, 1) > 0) {
            switch (c) {
                case 'p':
                case 'P':
                    is_paused.store(!is_paused.load());
                    if (is_paused.load()) {
                        std::cout << "\n[暂停] 按 'p' 继续，按 'q' 退出" << std::endl;
                    } else {
                        std::cout << "\n[继续] 按 'p' 暂停，按 'q' 退出" << std::endl;
                    }
                    break;
                case 'q':
                case 'Q':
                    should_exit.store(true);
                    std::cout << "\n[退出] 正在安全退出..." << std::endl;
                    break;
                case 's':
                case 'S':
                    std::cout << "\n[状态] 当前状态: " << (is_paused.load() ? "暂停" : "运行") << std::endl;
                    break;
                case 'h':
                case 'H':
                    std::cout << "\n[帮助] 按键说明:" << std::endl;
                    std::cout << "  p/P - 暂停/继续" << std::endl;
                    std::cout << "  q/Q - 退出程序" << std::endl;
                    std::cout << "  s/S - 显示状态" << std::endl;
                    std::cout << "  h/H - 显示帮助" << std::endl;
                    break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // 恢复终端设置
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

// Configuration file path finder
inline std::string GetMainConfigPath() {
    // Try relative path
    std::string relative_path = "config.yaml";
    if (std::filesystem::exists(relative_path)) {
        return relative_path;
    }
    
    // Try parent directory
    std::string parent_path = "../config.yaml";
    if (std::filesystem::exists(parent_path)) {
        return parent_path;
    }
    
    // Use absolute path as fallback
    return "/home/watermango/github/slam/config.yaml";
}

int main(int argc, char** argv) {
    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        config_path = GetMainConfigPath();
    }
    // Load main configuration
    std::cout << "Loading configuration from: " << config_path << std::endl;
    
    MainConfig config;
    if (!LoadMainConfiguration(config_path, config)) {
        std::cerr << "Failed to load configuration file" << std::endl;
        return 1;
    }
    
    std::cout << "Configuration loaded successfully" << std::endl;
    
    // Load calibration data
    std::cout << "Loading calibration data from: " << config.calibration_config_path << std::endl;
    CalibrationData calibration_data;
    if (!LoadCalibrationConfiguration(config.calibration_config_path, calibration_data)) {
        std::cerr << "Failed to load calibration data" << std::endl;
        return 1;
    }
    
    if (config.enable_debug) {
        std::cout << "Calibration data loaded:" << std::endl;
        std::cout << "  Car ID: " << calibration_data.car_id << std::endl;
        std::cout << "  Version: " << calibration_data.version << std::endl;
        std::cout << "  Camera: " << calibration_data.intrinsic_camera.camera_name << std::endl;
        std::cout << "  Resolution: " << calibration_data.intrinsic_camera.width << "x" << calibration_data.intrinsic_camera.height << std::endl;
    }
    
    // Load image data
    // RawImageData target_image;
    // std::vector<RawImageData> query_images;
    // LoadRawImageData(config.target_image_id, config.query_image_start_id, 
    //                  config.query_image_length, target_image, query_images);
    
    // if (config.enable_debug) {
    //     std::cout << "Loaded " << query_images.size() << " query images" << std::endl;
    //     std::cout << "Target image ID: " << target_image.id << std::endl;
    // }

    using namespace bagio;

    TxtDataLoader::Options options;
    options.gnss_file = config.gnss_file;
    options.imu_file = config.imu_file;
    options.odom_file = config.odom_file;

    TxtDataLoader data_loader(options);
    data_loader.load();

    std::cout << "Loaded counts: imu=" << data_loader.imuCount()
              << " odom=" << data_loader.odomCount()
              << " gnss=" << data_loader.gnssCount() << "\n";
    std::cout << "Time range: [" << std::to_string(data_loader.startTime())
              << ", " << std::to_string(data_loader.endTime())  
              << "]  duration=" << data_loader.duration() << " s\n";
              
    auto qi = data_loader.imuQueue();
    auto qo = data_loader.odomQueue();

    // 使用配置文件创建ESKF实例
    Eigen::Isometry3d T_iw = Eigen::Isometry3d(calibration_data.extrinsic_body_T_wheel.transform.inverse());
    Eigen::Quaterniond q_iw(T_iw.rotation());
    Eigen::Vector3d t_iw = T_iw.translation();

    std::cout << "Creating ESKF from configuration file..." << std::endl;
    eskf filter = eskf::fromConfigFile(config_path, q_iw, t_iw);
    std::cout << "ESKF created successfully with parameters from config file" << std::endl;

    std::priority_queue<Event, std::vector<Event>, CmpEvent> pq;
    if (!qi.empty()) { auto m = qi.front(); qi.pop(); pq.push({Event::IMU,  m->timestamp, m, {}}); }
    if (!qo.empty()) { auto m = qo.front(); qo.pop(); pq.push({Event::ODOM, m->timestamp, {}, m}); }

    
    std::string out_path = "eskf_result.txt";
    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "无法写入文件: " << out_path << "\n";
        return 1;
    }
    fout << std::fixed << std::setprecision(9);
    // 表头：时间 位姿 速度 四元数
    fout << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";

    // 4) 驱动滤波
    double last_progress_time = 0.0;
    double progress_interval = 1.0; // 每1秒打印一次进度
    int processed_count = 0;
    int total_events = qi.size() + qo.size();
    
    std::cout << "开始处理数据，总共 " << total_events << " 个事件（传感器数据）..." << std::endl;
    std::cout << "按键控制: p-暂停/继续, q-退出, s-状态, h-帮助" << std::endl;
    
    // 启动键盘输入处理线程
    std::thread keyboard_thread(keyboardInputHandler);
    keyboard_thread.detach();
    
    while (!pq.empty() && !should_exit.load()) {
        // 检查暂停状态
        while (is_paused.load() && !should_exit.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // 如果用户要求退出，跳出循环
        if (should_exit.load()) {
            break;
        }
        Event ev = pq.top(); pq.pop();
        processed_count++;
        
        // push 下一条
        if (ev.type == Event::IMU  && !qi.empty()) { auto m = qi.front(); qi.pop(); pq.push({Event::IMU,  m->timestamp, m, {}}); }
        if (ev.type == Event::ODOM && !qo.empty()) { auto m = qo.front(); qo.pop(); pq.push({Event::ODOM, m->timestamp, {}, m}); }

        if (ev.type == Event::IMU) {
            auto m = ev.imu;
            // // IMU 原始量（注意：这里默认 IMU 数据在 IMU系 i）
            Eigen::Vector3d gyroscope_raw(m->gx, m->gy, m->gz);
            Eigen::Vector3d accelerometer_raw(m->ax, m->ay, m->az);
            filter.feedimu(m->timestamp, accelerometer_raw, gyroscope_raw);
        } else {
            auto m = ev.odom;
            // 轮速：使用 twist.linear.x 作为前向速度（与 wheel-x 对齐）
            // filter.updateWheelSpeed(m->timestamp, m->vx);
            filter.feedwheelvelocity(m->timestamp, m->vx);
        }

        // 定期打印进度和当前状态
        if (ev.t - last_progress_time >= progress_interval) {
            double progress_percent = (double)processed_count / total_events * 100.0;
            std::cout << "进度: " << std::fixed << std::setprecision(1) << progress_percent 
                      << "% (" << processed_count << "/" << total_events << ")";
            
            const auto& S = filter.getNominalState();
            if (S.initialized) {
                std::cout << " | 位置: [" << std::setprecision(3) 
                          << S.p.x() << ", " << S.p.y() << ", " << S.p.z() << "]"
                          << " | 速度: [" << std::setprecision(3)
                          << S.v.x() << ", " << S.v.y() << ", " << S.v.z() << "]";
            } else {
                std::cout << " | 状态: 未初始化";
            }
            
            // 显示暂停状态
            if (is_paused.load()) {
                std::cout << " | [暂停中]";
            }
            
            std::cout << std::endl;
            last_progress_time = ev.t;
        }

        const auto& S = filter.getNominalState();
        if (S.initialized) {
            const auto& S = filter.getNominalState();
            fout << S.timestamp << ","
                     << S.p.x() << "," << S.p.y() << "," << S.p.z() << ","
                     << S.v.x() << "," << S.v.y() << "," << S.v.z() << ","
                     << S.q.w() << "," << S.q.x() << "," << S.q.y() << "," << S.q.z()
                     << "\n";
        } else {
            // 未初始化期间不输出pose，这里可打印监控行（可选）
            // std::cout << ev.t << ",0,,,,,,,,,\n";
        }
    }

    fout.close();

    // 设置退出标志，等待键盘线程结束
    should_exit.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 输出最终状态
    if (should_exit.load()) {
        std::cout << "\n程序被用户中断，已处理 " << processed_count << " 个事件" << std::endl;
    } else {
        std::cout << "\n数据处理完成，共处理 " << processed_count << " 个事件" << std::endl;
    }

    // if (!filter.isInitialized()) {
    //     std::cerr << "警告：未检测到足够长的静止段，未完成初始化。\n";
    // } else {
    //     const auto& S = filter.getNominalState();
    //     std::cout << "# Final p: " << S.position.transpose() << "\n";
    //     std::cout << "# Final v: " << S.velocity.transpose() << "\n";
    // }

    std::cout << "结果已保存到: " << out_path << std::endl;
    return 0;
}