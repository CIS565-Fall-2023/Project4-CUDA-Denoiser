#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise, int filter_size, bool weighted, float c_phi, float n_phi, float p_phi);
void showGBuffer(uchar4 *pbo, bool showPosition);
void showImage(uchar4 *pbo, int iter, bool denoise);
