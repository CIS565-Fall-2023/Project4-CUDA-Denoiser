#pragma once

#include <vector>
#include "config.h"

enum RenderBufferType {
	COLOR,
	NORMAL, 
	POSITION,
	DENOISED
};

struct DenoiseInfo {
	int filter_size = 10;
	float c_weight = 0.45f;
	float n_weight = 0.35f;
	float p_weight = 0.2f;
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(SceneConfig *hst_scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, RenderBufferType renderBufferType, DenoiseInfo & info);
void pathtraceInitBeforeMainLoop(SceneConfig* config);
void pathtraceFreeAfterMainLoop();
