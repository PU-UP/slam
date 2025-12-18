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
#include "dr/dr_odo_flow.hpp"
#include "eskf/eskf.hpp"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <mutex>
#include <sstream>

#include <cstring>      // for std::memset
#include <filesystem>   // 你用到了 std::filesystem::exists

struct TrackPose {
    double x{0.0}, y{0.0};
    double yaw{0.0};   // rad
    int init{0};
};

struct TelemetryStateMulti {
    double t{0.0};     // 建议用“当前系统时间戳”（eskf时间或事件时间）
    int paused{0};
    TrackPose eskf;
    TrackPose dr;
    TrackPose gnss;
};

std::mutex g_state_mtx;
TelemetryStateMulti g_latest;

// 全局控制变量
std::atomic<bool> is_paused(false);
std::atomic<bool> should_exit(false);

// ====== yaw 计算：假设 ENU / z-up（如果你的坐标不同再调整） ======
inline double YawFromQuatENU(const Eigen::Quaterniond& q) {
    // yaw (Z axis)
    double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    return std::atan2(siny_cosp, cosy_cosp);
}

class UdpTelemetrySender {
    public:
        UdpTelemetrySender(const std::string& ip, int port, double hz)
            : ip_(ip), port_(port), period_ms_((int)(1000.0 / hz)) {}
    
        bool start() {
            sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_ < 0) return false;
    
            std::memset(&addr_, 0, sizeof(addr_));
            addr_.sin_family = AF_INET;
            addr_.sin_port = htons(port_);
            inet_pton(AF_INET, ip_.c_str(), &addr_.sin_addr);
    
            th_ = std::thread([this](){ this->loop(); });
            th_.detach();
            return true;
        }
    
    private:
        void loop() {
            while (!should_exit.load()) {
                TelemetryStateMulti s;
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    s = g_latest;
                }
    
                // 简单 JSON（手写拼接，够用）
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << std::setprecision(6);

                oss << "{"
                    << "\"t\":" << s.t
                    << ",\"paused\":" << s.paused
                    << ",\"eskf\":{"
                        << "\"init\":" << s.eskf.init
                        << ",\"x\":" << s.eskf.x
                        << ",\"y\":" << s.eskf.y
                        << ",\"yaw\":" << s.eskf.yaw
                    << "}"
                    << ",\"dr\":{"
                        << "\"init\":" << s.dr.init
                        << ",\"x\":" << s.dr.x
                        << ",\"y\":" << s.dr.y
                        << ",\"yaw\":" << s.dr.yaw
                    << "}"
                    << ",\"gnss\":{"
                        << "\"init\":" << s.gnss.init
                        << ",\"x\":" << s.gnss.x
                        << ",\"y\":" << s.gnss.y
                        // GNSS没有航向也可以发 yaw=0；如果你更想省字段就删掉这一行
                        << ",\"yaw\":" << s.gnss.yaw
                    << "}"
                    << "}";

                std::string msg = oss.str();
                ::sendto(sock_, msg.data(), msg.size(), 0,
                        (struct sockaddr*)&addr_, sizeof(addr_));

    
                std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
            }
            ::close(sock_);
        }
    
    private:
        std::string ip_;
        int port_;
        int period_ms_;
        int sock_{-1};
        sockaddr_in addr_{};
        std::thread th_;
};

class UdpControlReceiver {
    public:
        UdpControlReceiver(int listen_port) : port_(listen_port) {}
    
        bool start() {
            sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_ < 0) return false;
    
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons(port_);
    
            if (bind(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                ::close(sock_);
                return false;
            }
    
            th_ = std::thread([this](){ this->loop(); });
            th_.detach();
            return true;
        }
    
    private:
        void loop() {
            char buf[1024];
            while (!should_exit.load()) {
                sockaddr_in src{};
                socklen_t slen = sizeof(src);
                int n = ::recvfrom(sock_, buf, sizeof(buf)-1, 0, (sockaddr*)&src, &slen);
                if (n <= 0) continue;
                buf[n] = '\0';
                std::string msg(buf);
    
                // 超轻量解析：只做 contains
                if (msg.find("\"cmd\":\"quit\"") != std::string::npos) {
                    should_exit.store(true);
                    is_paused.store(false);
                    std::cout << "\n[UDP] quit\n";
                } else if (msg.find("\"cmd\":\"pause\"") != std::string::npos) {
                    is_paused.store(true);
                    std::cout << "\n[UDP] pause\n";
                } else if (msg.find("\"cmd\":\"resume\"") != std::string::npos) {
                    is_paused.store(false);
                    std::cout << "\n[UDP] resume\n";
                } else if (msg.find("\"cmd\":\"toggle_pause\"") != std::string::npos) {
                    is_paused.store(!is_paused.load());
                    std::cout << "\n[UDP] toggle_pause -> " << (is_paused.load()?"paused":"running") << "\n";
                }
            }
            ::close(sock_);
        }
    
    private:
        int port_;
        int sock_{-1};
        std::thread th_;
};
    

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
    auto qg = data_loader.gnssQueue();

    // 使用配置文件创建ESKF实例
    Eigen::Isometry3d T_iw = Eigen::Isometry3d(calibration_data.extrinsic_body_T_wheel.transform);
    Eigen::Quaterniond q_iw(T_iw.rotation());
    Eigen::Vector3d t_iw = T_iw.translation();

    std::cout << "Creating ESKF from configuration file..." << std::endl;
    eskf filter = eskf::fromConfigFile(config_path, q_iw, t_iw);
    std::cout << "ESKF created successfully with parameters from config file" << std::endl;

    // 创建DR实例
    std::cout << "Creating DR (Dead Reckoning) instance..." << std::endl;
    dr_odom::DrOdoFlow dr_filter(config_path, calibration_data, config.dr_debug);
    std::cout << "DR created successfully" << std::endl;

    std::priority_queue<Event, std::vector<Event>, CmpEvent> pq;
    if (!qi.empty()) { auto m = qi.front(); qi.pop(); pq.push({Event::IMU,  m->timestamp, m, {}, {}}); }
    if (!qo.empty()) { auto m = qo.front(); qo.pop(); pq.push({Event::ODOM, m->timestamp, {}, m, {}}); }
    if (!qg.empty()) { auto m = qg.front(); qg.pop(); pq.push({Event::GNSS, m->timestamp, {}, {}, m}); }

    
    std::string eskf_out_path = "eskf_result.txt";
    std::ofstream eskf_fout(eskf_out_path);
    if (!eskf_fout) {
        std::cerr << "无法写入ESKF文件: " << eskf_out_path << "\n";
        return 1;
    }
    eskf_fout << std::fixed << std::setprecision(9);
    // TUM格式：timestamp tx ty tz qx qy qz qw（无表头）

    std::string dr_out_path = "dr_result.txt";
    std::ofstream dr_fout(dr_out_path);
    if (!dr_fout) {
        std::cerr << "无法写入DR文件: " << dr_out_path << "\n";
        return 1;
    }
    dr_fout << std::fixed << std::setprecision(9);
    // TUM格式：timestamp tx ty tz qx qy qz qw（无表头）

    std::string gnss_out_path = "gnss_result.txt";
    std::ofstream gnss_fout(gnss_out_path);
    if (!gnss_fout) {
        std::cerr << "无法写入GNSS文件: " << gnss_out_path << "\n";
        return 1;
    }
    gnss_fout << std::fixed << std::setprecision(9);
    // TUM格式：timestamp tx ty tz qx qy qz qw（无表头，GNSS基于第一个点作为锚点）

    // 4) 驱动滤波
    double last_progress_time = 0.0;
    double progress_interval = 1.0; // 每1秒打印一次进度
    int processed_count = 0;
    int total_events = qi.size() + qo.size() + qg.size();
    
    std::cout << "开始处理数据，总共 " << total_events << " 个事件（传感器数据）..." << std::endl;
    std::cout << "按键控制: p-暂停/继续, q-退出, s-状态, h-帮助" << std::endl;
    
    // 启动键盘输入处理线程
    std::thread keyboard_thread(keyboardInputHandler);
    keyboard_thread.detach();

    // 端口约定：
    // C++ -> Python telemetry: 19001
    // Python -> C++ control:   19002
    UdpTelemetrySender telemetry("127.0.0.1", 19001, 20.0); // 20Hz
    if (!telemetry.start()) {
        std::cerr << "Telemetry UDP start failed\n";
    }
    UdpControlReceiver control(19002);
    if (!control.start()) {
        std::cerr << "Control UDP start failed\n";
    }

    
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
        if (ev.type == Event::IMU  && !qi.empty()) { auto m = qi.front(); qi.pop(); pq.push({Event::IMU,  m->timestamp, m, {}, {}}); }
        if (ev.type == Event::ODOM && !qo.empty()) { auto m = qo.front(); qo.pop(); pq.push({Event::ODOM, m->timestamp, {}, m, {}}); }
        if (ev.type == Event::GNSS && !qg.empty()) { auto m = qg.front(); qg.pop(); pq.push({Event::GNSS, m->timestamp, {}, {}, m}); }

        if (ev.type == Event::IMU) {
            auto m = ev.imu;
            // IMU 原始量（注意：这里默认 IMU 数据在 IMU系 i）
            Eigen::Vector3d gyroscope_raw(m->gx, m->gy, m->gz);
            Eigen::Vector3d accelerometer_raw(m->ax, m->ay, m->az);
            
            // 同时处理ESKF和DR
            filter.feedimu(m->timestamp, accelerometer_raw, gyroscope_raw);
            dr_filter.setImu(m->timestamp, m->ax, m->ay, m->az, m->gx, m->gy, m->gz);
        } else if (ev.type == Event::ODOM) {
            auto m = ev.odom;
            // 轮速：使用 twist.linear.x 作为前向速度（与 wheel-x 对齐）
            filter.feedwheelvelocity(m->timestamp, m->vx);
            dr_filter.setWheel(m->timestamp, m->vx, m->wz);
        } else if (ev.type == Event::GNSS) {
            // GNSS处理：使用filter处理GNSS数据并转换到ENU坐标系
            auto m = ev.gnss;
            
            // 将GNSS数据传入filter（会自动设置锚点并转换到ENU）
            filter.feedRawGNSS(m->timestamp, m->lat, m->lon, m->alt);
            
            // 从filter获取ENU坐标
            double enu_e, enu_n, enu_u;
            if (filter.getENUPosition(enu_e, enu_n, enu_u)) {
                // 保存GNSS轨迹到文件（TUM格式：timestamp tx ty tz qx qy qz qw）
                // 注意：GNSS只有位置信息，四元数设为单位四元数
                gnss_fout << m->timestamp << " "
                         << enu_e << " " << enu_n << " " << enu_u << " "
                         << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0
                         << "\n";
                
                // 更新GNSS轨迹到全局状态
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_latest.gnss.x = enu_e;
                    g_latest.gnss.y = enu_n;
                    g_latest.gnss.yaw = 0.0;  // GNSS没有航向信息
                    g_latest.gnss.init = 1;
                }
            }
        }
        
        // 运行DR处理（仅当处理IMU或ODOM时）
        if (ev.type == Event::IMU || ev.type == Event::ODOM) {
            dr_filter.Run();
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

        // 保存ESKF结果（仅当处理IMU或ODOM时）
        if (ev.type == Event::IMU || ev.type == Event::ODOM) {
            // 先获取ESKF状态
            const auto& S = filter.getNominalState();
            
            // 再获取DR状态
            double dr_time;
            Eigen::Matrix4d dr_pose;
            dr_filter.getPose(dr_time, dr_pose);
            
            // 构建完整的TelemetryStateMulti对象
            TelemetryStateMulti ts;
            ts.paused = is_paused.load();
            
            // 先复制上一次的GNSS数据（如果GNSS本次未更新，保留上一次的值）
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                ts.gnss = g_latest.gnss;
            }
            
            // 设置ESKF轨迹
            if (S.initialized) {
                ts.t = S.timestamp;
                ts.eskf.x = S.p.x();
                ts.eskf.y = S.p.y();
                ts.eskf.yaw = YawFromQuatENU(S.q);
                ts.eskf.init = 1;
                
                // TUM格式：timestamp tx ty tz qx qy qz qw
                eskf_fout << S.timestamp << " "
                         << S.p.x() << " " << S.p.y() << " " << S.p.z() << " "
                         << S.q.x() << " " << S.q.y() << " " << S.q.z() << " " << S.q.w()
                         << "\n";
            } else {
                ts.eskf.init = 0;
            }
            
            // 设置DR轨迹
            if (dr_time > 0) {
                Eigen::Vector3d dr_position = dr_pose.block<3,1>(0,3);
                Eigen::Matrix3d dr_rotation = dr_pose.block<3,3>(0,0);
                Eigen::Quaterniond dr_quat(dr_rotation);
                
                ts.dr.x = dr_position.x();
                ts.dr.y = dr_position.y();
                ts.dr.yaw = YawFromQuatENU(dr_quat);
                ts.dr.init = 1;
                
                // TUM格式：timestamp tx ty tz qx qy qz qw
                dr_fout << dr_time << " "
                       << dr_position.x() << " " << dr_position.y() << " " << dr_position.z() << " "
                       << dr_quat.x() << " " << dr_quat.y() << " " << dr_quat.z() << " " << dr_quat.w()
                       << "\n";
            } else {
                ts.dr.init = 0;
            }
            
            // 最后一次性更新全局状态
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                g_latest = ts;
            }
        }
    }

    eskf_fout.close();
    dr_fout.close();
    gnss_fout.close();

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

    std::cout << "ESKF结果已保存到: " << eskf_out_path << std::endl;
    std::cout << "DR结果已保存到: " << dr_out_path << std::endl;
    std::cout << "GNSS结果已保存到: " << gnss_out_path << std::endl;
    return 0;
}