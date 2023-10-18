#pragma once

#include <thrust/device_ptr.h>
#include <vector>
#include "sceneStructs.h"

class GPUScene;
class Scene;

struct DenoiseConfig
{
	bool	denoise = false;
	int		level = 3;
	float	colorWeight = 0.45f;
	float	normalWeight = 0.35f;
	float	positionWeight = 0.2f;
};

class CudaPathTracer
{
public:
	CPU_ONLY CudaPathTracer() {}
	CPU_ONLY ~CudaPathTracer();

	CPU_ONLY void Resize(const int& w, const int& h);
	CPU_ONLY void Init(Scene* scene);
	CPU_ONLY void Reset() { m_Iteration = 1; }
	CPU_ONLY void GetImage(uchar4* host_image);
	CPU_ONLY void RegisterPBO(unsigned int pbo);
	CPU_ONLY void UnRegisterPBO() 
	{
		cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0);
		cudaGraphicsUnregisterResource(cuda_pbo_dest_resource);
	}

	CPU_ONLY void Render(GPUScene& scene, 
						const Camera& camera,
						const UniformMaterialData& data);
public:
	int m_Iteration;
	int m_MaxIteration = 1;

	glm::vec3* dev_hdr_img = nullptr; // device image that map color channels to [0, inf]
	uchar4* dev_img = nullptr; // device image that map color channels to [0, 255]

	PathSegment* dev_paths = nullptr;
	PathSegment* dev_end_paths = nullptr;
	ShadeableIntersection* dev_intersections = nullptr;
	GInfo* dev_gbuffer = nullptr;

	glm::vec3* dev_denoised_img_r;
	glm::vec3* dev_denoised_img_w;

	struct cudaGraphicsResource* cuda_pbo_dest_resource = nullptr;

	thrust::device_ptr<PathSegment> thrust_dev_paths_begin;
	thrust::device_ptr<PathSegment> thrust_dev_end_paths_bgein;

	glm::ivec2 resolution;
	DisplayMode m_DisplayMode = DisplayMode::Color;
	
	DenoiseConfig m_DenoiseConfig;
};