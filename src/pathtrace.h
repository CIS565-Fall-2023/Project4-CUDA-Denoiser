#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showDenoised(uchar4* pbo, int f_size, float c_phi, float n_phi, float p_phi, int iter);
void showImage(uchar4 *pbo, int iter);
