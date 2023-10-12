#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void clearGBuffer();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, bool showNormals, int iter);
void showImage(uchar4 *pbo, int iter);
