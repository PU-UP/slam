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
# 查找yaml-cpp
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# 打印找到的包信息
message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "yaml-cpp found: ${YAML_CPP_VERSION}")
# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
    pthread
)


