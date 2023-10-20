#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int frame, int iteration);
void showGBuffer(uchar4 *pbo, GBufferType gBufType);
void showImage(uchar4 *pbo, int iter);
void denoiseImage(uchar4* pbo, int iter, int filterSize, float colorWeight, float normalWeight, float positionWeight);
