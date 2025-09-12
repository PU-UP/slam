# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses CMake with C++17 standard. The main build commands are:

```bash
# Create build directory and configure
mkdir build && cd build
cmake ..

# Build the project
make -j$(nproc)

# Available executables:
# - app: Main application (data loading demonstration)
# - pipeline_demo: Visual SLAM pipeline demo with real images
# - test_feature_extractor: Feature extraction module test
# - test_feature_tracker: Feature tracking module test
```

## Dependencies

The project requires:
- **Eigen3**: Linear algebra library (version 3.3+)
- **OpenCV**: Computer vision library (version 4.5+) for image processing and feature detection
- **Ceres Solver**: Non-linear optimization library (version 2.0+) for bundle adjustment
- **yaml-cpp**: YAML configuration file parsing (via pkg-config)

## Project Architecture

This is a cleaned SLAM (Simultaneous Localization and Mapping) practice project focused on core visual pipeline components:

### Core Libraries

1. **data_prepare** (`data_prepare.cpp/.hpp`): 
   - Handles calibration data loading from YAML configuration
   - Manages camera, IMU, and wheel sensor parameters
   - Provides data structures for sensor calibration and raw image data
   - Uses Eigen for matrix operations and yaml-cpp for configuration parsing

2. **modules** (in `src/` directory):
   - **Feature Extractors** (`feature_extractors.cpp`): ORB feature detection
   - **Feature Trackers** (`feature_trackers.cpp`): Lucas-Kanade optical flow tracking
   - **Depth Estimators** (`depth_estimators.cpp`): Triangulation and 3D point reconstruction
   - **Bundle Adjusters** (`bundle_adjusters.cpp`): Ceres-based pose and point optimization

### Applications

- **main.cpp**: Demonstrates configuration and calibration data loading
- **pipeline_demo.cpp**: Complete visual SLAM pipeline demonstration with real images
  - Loads real images from `/home/watermango/data/raw_data_for_loop_closure/image/`
  - Demonstrates feature extraction, tracking, depth estimation, and bundle adjustment
  - Includes visualization for all pipeline stages

### Configuration

- **config.yaml**: Main configuration with image paths and debug options
- **calibration_config.yaml**: Contains camera intrinsics, IMU parameters, wheel parameters, and extrinsic transformations between sensors
- The configuration includes transformations between body, camera, wheel, and RTK coordinate frames

### Key Data Structures

- `CalibrationData`: Complete sensor calibration information
- `CameraParams`: Pinhole camera model with distortion parameters
- `RawImageData`: Image with associated wheel pose and timestamp
- `FeatureExtractor::FeaturesResult`: Feature extraction results
- `FeatureTracker::TrackingResult`: Feature tracking results
- `DepthEstimator::DepthResult`: 3D triangulation results
- `BundleAdjuster::BAResult`: Bundle adjustment optimization results

### Visual SLAM Pipeline

The complete pipeline follows this flow:
1. **Configuration Loading**: Load main config and calibration data
2. **Feature Extraction**: ORB feature detection on first frame
3. **Feature Tracking**: LK optical flow tracking between frames
4. **Depth Estimation**: 3D point triangulation using wheel-to-camera poses
5. **Bundle Adjustment**: Joint optimization of poses and 3D points
6. **Visualization**: Display results for each pipeline stage

### Key Technical Details

- **Coordinate Transformations**: Uses wheel-to-camera extrinsic transformation from calibration data
- **Real Image Processing**: Processes actual images from specified dataset paths
- **Comprehensive Visualization**: Real-time display of features, tracks, 3D points, and reprojection errors
- **Performance Metrics**: Reports processing time and accuracy metrics for each stage

## Common Issues

- All calibration data (K, dist, T_wheel_cam) is properly loaded from configuration files
- Feature tracking uses LK optical flow with geometric consistency checks
- Bundle adjustment uses Ceres Solver with robust loss functions
- Pipeline demo requires access to the specified image paths
- Visualization windows may require manual closing due to waitKey calls