#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, const bool& ui_showGbuffer, 
	const bool& ui_showGbufferNormal, const bool& ui_showGbufferPos);
void showImage(uchar4 *pbo, int iter);
