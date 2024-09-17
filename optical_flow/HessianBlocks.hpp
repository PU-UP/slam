/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#define MAX_ACTIVE_FRAMES 100

 
#include "vector"
 
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>


#define PYR_LEVELS 6
extern int pyrLevelsUsed;
extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		cxG[PYR_LEVELS], cyG[PYR_LEVELS];

extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

extern Eigen::Matrix3f KG[PYR_LEVELS],KiG[PYR_LEVELS];

struct FrameHessian;


#define SCALE_IDEPTH 1.0f		// scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)


struct FrameHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	Eigen::Vector3f* dI;				 // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
	Eigen::Vector3f* dIp[PYR_LEVELS];	 // coarse tracking / coarse initializer. NAN in [0] only.
	float* absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.
	std::vector<Eigen::Vector2f> keypoints;


	void release();

	inline ~FrameHessian()
	{
		release(); 
		for(int i=0;i<pyrLevelsUsed;i++)
		{
			delete[] dIp[i];
			delete[]  absSquaredGrad[i];

		}
	};
	inline FrameHessian()
	{
	};


    void makeImages(float* color);

};







