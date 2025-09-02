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

# The main executable is named 'app'
```

## Dependencies

The project requires:
- **Eigen3**: Linear algebra library (version 3.3+)
- **OpenCV**: Computer vision library (version 4.5+) for image processing and feature detection
- **Ceres Solver**: Non-linear optimization library (version 2.0+) for bundle adjustment
- **yaml-cpp**: YAML configuration file parsing (via pkg-config)

## Project Architecture

This is a SLAM (Simultaneous Localization and Mapping) practice project with two main components:

### Core Libraries

1. **data_prepare** (`data_prepare.cpp/.hpp`): 
   - Handles calibration data loading from YAML configuration
   - Manages camera, IMU, and wheel sensor parameters
   - Provides data structures for sensor calibration and raw image data
   - Uses Eigen for matrix operations and yaml-cpp for configuration parsing

2. **sfm_reconstructor** (`sfm_reconstructor.cpp/.hpp`):
   - Implements incremental Structure from Motion (SfM) with visual odometry
   - Uses ORB feature detection and LK optical flow for tracking
   - Performs bundle adjustment with Ceres Solver including wheel odometry priors
   - Exports results to CSV/JSON for visualization

### Main Application

- **main.cpp**: Demonstrates usage by loading calibration data and running SfM reconstruction

### Configuration

- **calibration_config.yaml**: Contains camera intrinsics, IMU parameters, wheel parameters, and extrinsic transformations between sensors
- The configuration includes transformations between body, camera, wheel, and RTK coordinate frames

### Key Data Structures

- `CalibrationData`: Complete sensor calibration information
- `CameraParams`: Pinhole camera model with distortion parameters
- `RawImageData`: Image with associated wheel pose and timestamp
- `SFMResult`: Output containing camera poses and 3D points
- `Frame`: Internal representation for SfM processing

### SfM Pipeline

The reconstruction pipeline follows this flow:
1. Load calibration data and extract features
2. Track features between frames using optical flow
3. Refine poses with PnP when sufficient features are tracked
4. Triangulate new landmarks between keyframes
5. Run global bundle adjustment with wheel odometry priors
6. Export results for visualization

## Common Issues

- The main.cpp:37 references undefined variables `K`, `dist`, `T_wheel_cam` - these need to be extracted from calibration data
- Feature tracking uses nearest-neighbor matching which may need robustification for production use
- Bundle adjustment currently fixes the first frame pose to eliminate scale ambiguity