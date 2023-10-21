#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop(bool& configChanged);
