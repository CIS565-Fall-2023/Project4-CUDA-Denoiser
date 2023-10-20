#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(
    int frame, 
	int iter, 
	bool denoise, 
	int filterSize, 
	float sigma_rt, 
	float sigma_n, 
	float sigma_x);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
