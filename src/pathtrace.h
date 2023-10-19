#pragma once

#include <vector>
#include "config.h"

enum RenderBufferType {
	COLOR,
	NORMAL, 
	POSITION,
	DEPTH
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(SceneConfig *hst_scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, RenderBufferType renderBufferType);
void pathtraceInitBeforeMainLoop(SceneConfig* config);
void pathtraceFreeAfterMainLoop();
