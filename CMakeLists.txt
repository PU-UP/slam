cmake_minimum_required(VERSION 3.16)
project(slam VERSION 1.0.0 LANGUAGES CXX)

# 生成compile_commands.json用于智能跳转
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 查找必要的依赖包
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
# 查找yaml-cpp
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# 打印找到的包信息
message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "yaml-cpp found: ${YAML_CPP_VERSION}")
message(STATUS "Ceres found: ${Ceres_VERSION}")
# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 创建modules库
add_library(modules
    src/modules.cpp
    src/feature_extractors.cpp
    src/feature_trackers.cpp
    src/depth_estimators.cpp
    src/bundle_adjusters.cpp
)

# 设置modules库的包含目录
target_include_directories(modules PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# 链接modules库的依赖
target_link_libraries(modules PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    stdc++fs
    ceres
)

# 创建data_prepare库
add_library(data_prepare data_prepare.cpp data_prepare.hpp)

# 创建dr_odo_flow库
add_library(dr_odo_flow 
    src/dr_odo_flow.cpp
    include/dr/dr_odo_flow.hpp
)

# 设置data_prepare库的包含目录
target_include_directories(data_prepare PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# 链接data_prepare库的依赖
target_link_libraries(data_prepare PUBLIC
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    Eigen3::Eigen
    stdc++fs
)

# 设置dr_odo_flow库的包含目录
target_include_directories(dr_odo_flow PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# 链接dr_odo_flow库的依赖
target_link_libraries(dr_odo_flow PUBLIC
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    Eigen3::Eigen
    stdc++fs
)



# 创建测试可执行文件
add_executable(test_feature_extractor test/test_feature_extractor.cpp)
add_executable(test_feature_tracker test/test_feature_tracker.cpp)

# 链接测试可执行文件的依赖
target_link_libraries(test_feature_extractor PRIVATE
    modules
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_link_libraries(test_feature_tracker PRIVATE
    modules
    ${OpenCV_LIBS}
    Eigen3::Eigen
)


# 创建主可执行文件
add_executable(app main.cpp)

# 设置slam可执行文件的包含目录
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接slam可执行文件的依赖
target_link_libraries(app PRIVATE
    data_prepare
    dr_odo_flow
    modules
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)

# 创建pipeline演示可执行文件
add_executable(pipeline_demo pipeline_demo.cpp)

# 设置pipeline_demo可执行文件的包含目录
target_include_directories(pipeline_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接pipeline_demo可执行文件的依赖
target_link_libraries(pipeline_demo PRIVATE
    data_prepare
    modules
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)


