#pragma once

#include "HessianBlocks.hpp"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class OpticalFlowTrackerDSO { 
  public:
    OpticalFlowTrackerDSO() {}

    void track(FrameHessian* cur_fh);

  private:
    FrameHessian* last_fh_ = nullptr;


};