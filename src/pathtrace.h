#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoisImage(uchar4 *pbo,int ui_filterSize, float ui_colorWeight,float ui_normalWeight,float ui_positionWeight,int iter,int level);
