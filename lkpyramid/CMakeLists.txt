cmake_minimum_required(VERSION 3.10)
project(OpticalFlowProject)

set(CMAKE_BUILD_TYPE "Debug")

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV 4 REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS} .)

# 源文件列表
set(SOURCES
    main.cpp
    lkpyramid.cpp  # 替换为你的其他源文件
    # 其他源文件
)

# 创建可执行文件
add_executable(OpticalFlow ${SOURCES})

# 链接OpenCV库
target_link_libraries(OpticalFlow ${OpenCV_LIBS})
