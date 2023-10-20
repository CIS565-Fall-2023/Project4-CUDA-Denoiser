#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, int interation, bool showGbuffer, bool showNormal);
void showImage(uchar4 *pbo, int iter);

void initDenoiser();
void denoiserFree();
void denoiser(glm::ivec2 resolution, int iter);
