#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, int filter, float c_phi, float n_phi, float p_phi, bool ui_denoize);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter, bool ui_denoize);
