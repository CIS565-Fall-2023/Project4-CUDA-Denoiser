#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, const DenoiserParameters& denoiserParams);
void showGBuffer(uchar4 *pbo, RenderMode renderMode);
void showImage(uchar4 *pbo, int iter);
