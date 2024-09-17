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


 
#include "HessianBlocks.hpp"

int pyrLevelsUsed = 6;

int wG[PYR_LEVELS], hG[PYR_LEVELS];
float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		cxG[PYR_LEVELS], cyG[PYR_LEVELS];

float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];


// wG[0] = 752;
// hG[0] = 480;
// KG[0] = K;
// fxG[0] = K(0,0);
// fyG[0] = K(1,1);
// cxG[0] = K(0,2);
// cyG[0] = K(1,2);
// KiG[0] = KG[0].inverse();
// fxiG[0] = KiG[0](0,0);
// fyiG[0] = KiG[0](1,1);
// cxiG[0] = KiG[0](0,2);
// cyiG[0] = KiG[0](1,2);

// for (int level = 1; level < pyrLevelsUsed; ++ level)
// {
// 	wG[level] = w >> level;
// 	hG[level] = h >> level;

// 	fxG[level] = fxG[level-1] * 0.5;
// 	fyG[level] = fyG[level-1] * 0.5;
// 	cxG[level] = (cxG[0] + 0.5) / ((int)1<<level) - 0.5;
// 	cyG[level] = (cyG[0] + 0.5) / ((int)1<<level) - 0.5;

// 	KG[level]  << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;	// synthetic
// 	KiG[level] = KG[level].inverse();

// 	fxiG[level] = KiG[level](0,0);
// 	fyiG[level] = KiG[level](1,1);
// 	cxiG[level] = KiG[level](0,2);
// 	cyiG[level] = KiG[level](1,2);
// }


void FrameHessian::release()
{
}


void FrameHessian::makeImages(float* color)
{
	int ww= 752, hh = 480;
	wG[0] = ww;
	hG[0] = hh;
	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		wG[level] = ww >> level;
		hG[level] = hh >> level;
	}

	for(int i=0;i<pyrLevelsUsed;i++)
	{
		std::cout << wG[i] << ", " << hG[i] << std::endl;
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	dI = dIp[0];


	// make d0
	int w=wG[0];
	int h=hG[0];
	// segmentation fault happens here
	for(int i=0;i<w*h;i++) {
		dI[i][0] = color[i];
	}

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl];
		Eigen::Vector3f* dI_l = dIp[lvl];

		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];



			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy;
		}
	}
}



