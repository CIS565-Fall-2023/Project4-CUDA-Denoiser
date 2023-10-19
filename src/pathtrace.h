#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBufferDepth(uchar4* pbo);
void showGBufferPos(uchar4* pbo);
void showGBufferNormal(uchar4* pbo);
void showDenoisedImage(uchar4* pbo, int iter, float colWeight, float norWeight, float posWeight, int filterSize);
void showImage(uchar4* pbo, int iter);
