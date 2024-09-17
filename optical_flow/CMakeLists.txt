cmake_minimum_required(VERSION 2.8)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)


# add_library(hessian SHARED HessianBlocks.cpp OpticalFlowTrackDSO.cpp)
add_library(hessian SHARED HessianBlocks.cpp)
add_library(optical_tracker OpticalFlowTrackDSO.cpp)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS} hessian optical_tracker)

# add_executable(direct_method direct_method.cpp)
# target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
