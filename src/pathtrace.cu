#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <numeric>
#include <cuda_runtime.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}


#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
static cudaEvent_t event_start = nullptr;
static cudaEvent_t event_end = nullptr;
float elapsed_time_gpu_ms = 0.f;
#endif


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__host__ __device__
glm::vec3 decodeNormal(float x, float y) {
	glm::vec3 nor = glm::vec3(x, y, 1.0f - abs(x) - abs(y));
	float t = glm::clamp(-nor.z, 0.0f, 1.0f);
	nor.x += nor.x >= 0.0f ? -t : t;
	nor.y += nor.y >= 0.0f ? -t : t;
	return glm::normalize(nor);
}

__host__ __device__
void encodeNormal(glm::vec3& nor) {
	float absSum = abs(nor.x) + abs(nor.y) + abs(nor.z);
	if (absSum > 0.001f) {
		nor /= absSum;
		if (nor.z < 0) {
			nor.x = (1.0f - abs(nor.y)) * (nor.x > 0.0f ? 1.0f : -1.0f);
			nor.y = (1.0f - abs(nor.x)) * (nor.y > 0.0f ? 1.0f : -1.0f);
		}
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}


__global__ void	gaussianBlur(glm::ivec2 resolution,
	int iter, glm::vec3* image, int filterSize) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		const float kernel[25] = {
			0.00391f, 0.01563f, 0.02344f, 0.01563f, 0.00391f,
			0.01563f, 0.0625f, 0.09375f, 0.0625f, 0.01563f,
			0.02344f, 0.09375f, 0.14063f,  0.09375f, 0.02344f,
			0.01563f, 0.0625f, 0.09375f, 0.0625f, 0.01563f,
			0.00391f, 0.01563f, 0.02344f, 0.01563f, 0.00391f
		};
		glm::vec3 sum = glm::vec3(0.0f);
		for (int dx = -2; dx <= 2; dx++) {
			for (int dy = -2; dy <= 2; dy++) {
				int nx = x + dx * filterSize;
				int ny = y + dy * filterSize;
				if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y) {
					int tmpIndex = nx + (ny * resolution.x);
					sum += image[tmpIndex] * kernel[dx + 2 + 5 * (dy + 2)];
				}
			}
		}

		image[index] = sum;
	}
}

__global__ void denoise(glm::ivec2 resolution, Camera cam,
	float colWeight, float norWeight, float posWeight, int step,
	int iter, glm::vec3* image, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		glm::vec3 sum = glm::vec3(0.);
		glm::vec3 col = image[index] / (float)iter;
#ifdef OCT_ENCODE_NORMAL
		glm::vec3 nor = decodeNormal(gBuffer[index].nx, gBuffer[index].ny);
#else
		glm::vec3 nor = gBuffer[index].normal;
#endif
		glm::vec3 dir = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
		glm::vec3 pos = gBuffer[index].t * dir + cam.position;
		// glm::vec3 pos = gBuffer[index].pos;

		//  0.0625, 0.25, 0.375, 0.25, 0.0625
		const float kernel[25] = {
			0.00391f, 0.01563f, 0.02344f, 0.01563f, 0.00391f,
			0.01563f, 0.0625f, 0.09375f, 0.0625f, 0.01563f,
			0.02344f, 0.09375f, 0.14063f,  0.09375f, 0.02344f,
			0.01563f, 0.0625f, 0.09375f, 0.0625f, 0.01563f,
			0.00391f, 0.01563f, 0.02344f, 0.01563f, 0.00391f
		};
		
		float sum_w = 0.;
		for (int dx = -2; dx <= 2; dx++) {
			for (int dy = -2; dy <= 2; dy++) {
				int nx = x + dx * step;
				int ny = y + dy * step;
				if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y) {
					int tmpIndex = nx + (ny * resolution.x);
					
					glm::vec3 tmpCol = image[tmpIndex] / (float)iter;
					glm::vec3 t = col - tmpCol;
					float dist2 = glm::dot(t, t);
					float cw = min(exp(-(dist2) / colWeight), 1.0f);

					
#ifdef OCT_ENCODE_NORMAL
					glm::vec3 tmpNor = decodeNormal(gBuffer[tmpIndex].nx, gBuffer[tmpIndex].ny);
#else
					glm::vec3 tmpNor = gBuffer[tmpIndex].normal;
#endif
					t = nor - tmpNor;
					dist2 = glm::dot(t, t);
					float nw = min(exp(-(dist2) / norWeight), 1.0f);

					// glm::vec3 tmpPos = gBuffer[tmpIndex].pos;
					glm::vec3 tmpDir = glm::normalize(cam.view
						- cam.right * cam.pixelLength.x * ((float)nx - (float)cam.resolution.x * 0.5f)
						- cam.up * cam.pixelLength.y * ((float)ny - (float)cam.resolution.y * 0.5f));
					glm::vec3 tmpPos = gBuffer[index].t * tmpDir + cam.position;
					t = pos - tmpPos;
					dist2 = glm::dot(t, t);
					float pw = min(exp(-(dist2) / posWeight), 1.0f);

					float w = cw * nw * pw;
					sum += tmpCol * w * kernel[dx + 2 + (dy + 2) * 5];
					sum_w += w * kernel[dx + 2 + (dy + 2) * 5];
				}
			}
		}
		col = sum / sum_w;

		image[index] = col * (float)iter;
	}
}


__global__ void gbufferDepthToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer[index].t * 256.0;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;
	}
}


__global__ void gbufferPosToPBO(uchar4* pbo, Camera cam, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 dir = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
		glm::vec3 pos = gBuffer[index].t * dir + cam.position;
		glm::ivec3 p = glm::clamp((glm::ivec3)(pos / 10.0f * 256.0f), glm::ivec3(0), glm::ivec3(255));

		pbo[index].w = 0;
		pbo[index].x = p.x;
		pbo[index].y = p.y;
		pbo[index].z = p.z;
	}
}


__global__ void gbufferNormalToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 nor = decodeNormal(gBuffer[index].nx, gBuffer[index].ny);
		nor = (nor * 0.5f + glm::vec3(0.5f)) * 256.0f;

		pbo[index].w = 0;
		pbo[index].x = nor.x;
		pbo[index].y = nor.y;
		pbo[index].z = nor.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static PathSegment* dev_tempPaths = NULL;
static PathSegment* dev_pathsBuffer = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_tempIntersections = NULL;
static ShadeableIntersection* dev_intersectionsBuffer = NULL;
static int* dev_bools = NULL;
static int* dev_nbools = NULL;
static int* dev_scanBools = NULL;
static int* dev_scanNBools = NULL;
static Triangle* dev_tris = NULL;
static BoundingBox* dev_bvh = NULL;
static GBufferPixel* dev_gBuffer = NULL;

void printArr(int n, int* odata, int* dev_odata) {
	cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_odata -> odata for printArr failed!");
	for (int i = 0; i <= n / 10; i++) {
		for (int j = 0; j < 10 && j < n - 10 * i; j++) {
			std::cout << odata[i * 10 + j] << "  ";
		}std::cout << std::endl;
	}std::cout << std::endl << std::endl;
}

void printArr(int begin, int n, int* dev_odata) {
	int o[10];
	for (int i = 0; i <= n / 10; i++) {
		cudaMemcpy(o, dev_odata + begin + 10 * i, sizeof(int) * 10, cudaMemcpyDeviceToHost);
		if (o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]+o[8]+o[9] == -10) {
			continue;
		}
		checkCUDAError("cudaMemcpy dev_odata -> odata for printArr failed!");
		for (int j = 0; j < 10 && j < n - 10 * i; j++) {
			std::cout << o[j] << "  ";
		}std::cout << std::endl;
	}std::cout << std::endl << std::endl;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {

	checkCUDAError("before pathtraceInit");

	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_tempPaths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_tempIntersections, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_bools, pixelcount * sizeof(int));
	cudaMalloc(&dev_nbools, pixelcount * sizeof(int));
	cudaMalloc(&dev_scanBools, pixelcount * sizeof(int));
	cudaMalloc(&dev_scanNBools, pixelcount * sizeof(int));

#ifdef CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_intersectionsBuffer, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_pathsBuffer, pixelcount * sizeof(PathSegment));
#endif

#ifdef USING_BVH
	cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(BoundingBox));
	cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(BoundingBox), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	checkCUDAError("pathtraceInit");


#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_end);

	checkCUDAError("pathtraceInit, cudaEventCreate");
	cudaEventRecord(event_start);
#endif
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	cudaFree(dev_bools);
	cudaFree(dev_nbools);
	cudaFree(dev_scanBools);
	cudaFree(dev_scanNBools);
	cudaFree(dev_tempPaths);
	cudaFree(dev_tempIntersections);
#ifdef CACHE_FIRST_BOUNCE
	cudaFree(dev_intersectionsBuffer);
	cudaFree(dev_pathsBuffer);
#endif
#ifdef USING_BVH
	cudaFree(dev_bvh);
#endif
	cudaFree(dev_gBuffer);

	checkCUDAError("pathtraceFree");

#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	if (event_start != nullptr) {
		cudaEventDestroy(event_start);
		cudaEventDestroy(event_end);
	}
#endif
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::random::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
#ifdef JITTER_RAY
		thrust::uniform_real_distribution<float> u1(-1, 1);
		thrust::uniform_real_distribution<float> u02PI(0, TWO_PI);
		float rz = sqrt(u1(rng));
		rz = ((rz > 0) ? 1 : -1) * sqrt(abs(rz));
#endif

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
#ifdef JITTER_RAY
			+ glm::normalize(glm::vec3(__sinf(u02PI(rng)), __cosf(u02PI(rng)), rz)) * JITTER_RATIO
#endif
		);


		// Depth of field
#ifdef DEPTH_OF_FIELD
		thrust::uniform_real_distribution<float> u01(-1, 1);
		glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
		glm::vec2 sampleDisk;
		if (length(sample) < 1e-5) { sampleDisk = glm::vec2(0, 0); }
		else {
			float theta, r;
			if (std::abs(sample.x) > std::abs(sample.y)) {
				r = sample.x;
				theta = (PI / 4.0) * (sample.y / sample.x);
			}
			else {
				r = sample.y;
				theta = (PI / 2.0) - (PI / 4.0) * (sample.x / sample.y);
			}
			sampleDisk = r * glm::vec2(std::cos(theta), std::sin(theta));
		}

		glm::vec2 pLens = DOF_LENS_RADIUS * sampleDisk;

		float ft = glm::abs(DOF_FOCAL_DISTANCE / segment.ray.direction.z);
		glm::vec3 pFocus = segment.ray.origin + ft * segment.ray.direction;

		segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif


		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.refractionBefore = false;
	}
}


// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersectionsNaive(
	int depth
	, int num_paths
	, const PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Triangle* tris
	, int tris_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_index = -1;
		bool hit_geom = true;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		for (int i = 0; i < tris_size; i++) {
			Triangle& tri = tris[i];
			t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_index = i;
				hit_geom = false;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_geom ? geoms[hit_index].materialid : tris[hit_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}




__global__ void computeIntersectionsBVH(
	int depth
	, int num_paths
	, const PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Triangle* tris
	, int tris_size
	, BoundingBox* bvh
	, int bvh_size
	, ShadeableIntersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = 1e5;
		int hit_index = -1;
		bool hit_geom = true;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}


		// __shared__ int arr[BLOCK_SIZE_1D][BVH_GPU_STACK_SIZE];
		int arr[BVH_GPU_STACK_SIZE];

		// int arr[BVH_GPU_STACK_SIZE];
		int sign = 0;
		arr[0] = 0;

		float o[3] = { pathSegment.ray.origin.x, pathSegment.ray.origin.y, pathSegment.ray.origin.z };

		glm::vec3 inv_dir = 1.0f / pathSegment.ray.direction;

		while (sign >= 0) {
			
			BoundingBox& bbox = bvh[arr[sign]];
			int beginId = bbox.beginTriId;
			sign--;


			float bbox_min_t = 1e-5;
			float bboX_max_t = 1e5;

			for (int a = 0; a < 3; a++) {

				float t0 = (bbox.min[a] - o[a]) * inv_dir[a];
				float t1 = (bbox.max[a] - o[a]) * inv_dir[a];

				bbox_min_t = fmax(min(t0, t1), bbox_min_t);
				bboX_max_t = fmin(max(t0, t1), bboX_max_t);
			}

			if(bboX_max_t >= bbox_min_t && bboX_max_t > 0.0f && bbox_min_t < t_min)
			{
				// reach leaf node of bvh
				if (beginId >= 0) {

					// TriangleArray& triIndices = tri_arr[taid];
					// #pragma unroll
					for (int j = beginId; j < bbox.triNum + beginId; j++) {

						t = triangleIntersectionTest(tris[j], pathSegment.ray, tmp_intersect, tmp_normal, outside);
						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_index = j;
							hit_geom = false;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
						}
					}
				}
				// keep searching
				else if (sign + 2 < BVH_GPU_STACK_SIZE) {
					arr[sign + 1] = bbox.leftId;
					arr[sign + 2] = bbox.rightId;
					sign += 2;
				}
			}
		}



		if (hit_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_geom ? geoms[hit_index].materialid : tris[hit_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}



// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, int depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			// thrust::uniform_real_distribution<float> u01(0, 1); // u01(rng) to get random (0, 1)

			Material material = materials[intersection.materialId];

			glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
			scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = -1;
		}
	}
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].t = shadeableIntersections[idx].t;
		// gBuffer[idx].pos = max(shadeableIntersections[idx].t, 0.0f) * glm::normalize(pathSegments[idx].ray.direction) + pathSegments[idx].ray.origin;
#ifdef OCT_ENCODE_NORMAL
		glm::vec3 nor = shadeableIntersections[idx].surfaceNormal;
		encodeNormal(nor);
		gBuffer[idx].nx = nor.x;
		gBuffer[idx].ny = nor.y;
#else
		gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
#endif
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color * (float)abs(iterationPath.remainingBounces);		
	}
}

__global__ void markPathSegment(int nPaths, int* bools, int* nbools, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		bool b = iterationPaths[index].remainingBounces > 0;
		bools[index] = b;
		nbools[index] = !b;
	}
}

// for path termination
__global__ void pathSegmentScatter(int n, int scanSum, PathSegment* odata, const PathSegment* idata, const int* bools, const int* indicesPos, const int* indicesNeg) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (bools[index] > 0) {
			odata[indicesPos[index]] = idata[index];
		}
		else {
			odata[indicesNeg[index] + scanSum] = idata[index];
		}
	}
}

// for material sort
__global__ void kernMapMatBitToBoolean(int n, int i, int* bools, int* ebools, const ShadeableIntersection* idata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		bools[index] = ((idata[index].materialId >> i) & 1);
		ebools[index] = !((idata[index].materialId >> i) & 1);
	}
}

// for path termination
__global__ void pathSegmentAndIntersectionScatter(
	int n, int negCount,
	PathSegment* opaths, const PathSegment* ipaths,
	ShadeableIntersection* ointers, const ShadeableIntersection* iinters,
	const int* bools, const int* indicesPos, const int* indicesNeg) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (bools[index] > 0) {
			opaths[indicesPos[index] + negCount] = ipaths[index];
			ointers[indicesPos[index] + negCount] = iinters[index];
		}
		else {
			opaths[indicesNeg[index]] = ipaths[index];
			ointers[indicesNeg[index]] = iinters[index];
		}
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
 ///////////////////////////////////////////////////////////////////////////

void pathtrace(int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = BLOCK_SIZE_1D;
	int depth = 0;
	int num_paths = pixelcount;

#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	cudaEventRecord(event_start);
#endif


#ifdef CACHE_FIRST_BOUNCE
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_pathsBuffer, dev_paths, sizeof(PathSegment) * pixelcount, cudaMemcpyDeviceToDevice);
		checkCUDAError("save dev_pathsBuffer");
	}
	else {
		cudaMemcpy(dev_paths, dev_pathsBuffer, sizeof(PathSegment) * pixelcount, cudaMemcpyDeviceToDevice);
		checkCUDAError("load dev_pathsBuffer");
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif


	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	int mat_num = hst_scene->materials.size();

	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		checkCUDAError("cudaMemset dev_intersections");

#ifdef CACHE_FIRST_BOUNCE
		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersectionsBuffer, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
			checkCUDAError("load dev_intersectionsBuffer");
		}
		else {
#endif
			// tracing
#ifdef USING_BVH
			if (hst_scene->bvh.size() > 0) {
				computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths,
					dev_geoms, hst_scene->geoms.size(),
					dev_tris, hst_scene->tris.size(),
					dev_bvh, hst_scene->bvh.size(),
					dev_intersections);
				checkCUDAError("tcomputeIntersectionsBVH");
			} else {
#endif
				computeIntersectionsNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths,
					dev_geoms, hst_scene->geoms.size(),
					dev_tris, hst_scene->tris.size(),
					dev_intersections);
#ifdef USING_BVH
			}
#endif


			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
#ifdef CACHE_FIRST_BOUNCE
			if (iter == 1 && depth == 0) {
				cudaMemcpy(dev_intersectionsBuffer, dev_intersections, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
				checkCUDAError("save dev_intersectionsBuffer");
			}
		}
#endif

		

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.


		// Sort All Materials
#ifdef MATERIAL_SORT
		if (mat_num > 1) {
			int log2Ceil = 1;
			int product = 1;
			while (product < mat_num) {
				product *= 2;
				log2Ceil++;
			}
			for (int i = 0; i < log2Ceil; i++) {
				kernMapMatBitToBoolean << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, i, dev_bools, dev_nbools, dev_intersections);
				thrust::device_ptr<int> dv_sort_in(dev_bools);
				thrust::device_ptr<int> dv_sort_out(dev_scanBools);
				thrust::exclusive_scan(dv_sort_in, dv_sort_in + num_paths, dv_sort_out);
				thrust::device_ptr<int> dv_sort_nin(dev_nbools);
				thrust::device_ptr<int> dv_sort_nout(dev_scanNBools);
				thrust::exclusive_scan(dv_sort_nin, dv_sort_nin + num_paths, dv_sort_nout);

				int neg_count = -1;
				cudaMemcpy(&neg_count, dev_scanNBools + num_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
				int last_bool = -1;
				cudaMemcpy(&last_bool, dev_bools + num_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
				neg_count += (last_bool == 0);

				pathSegmentAndIntersectionScatter << < numblocksPathSegmentTracing, blockSize1d >> > (
					num_paths, neg_count,
					dev_tempPaths, dev_paths,
					dev_tempIntersections, dev_intersections,
					dev_bools, dev_scanBools, dev_scanNBools);

				PathSegment* tempPath = dev_tempPaths;
				dev_tempPaths = dev_paths;
				dev_paths = tempPath;
				ShadeableIntersection* tempInter = dev_tempIntersections;
				dev_tempIntersections = dev_intersections;
				dev_intersections = tempInter;
			}

		}
#endif
		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		depth++;

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, depth,
			dev_intersections, dev_paths, dev_materials
		);
		
#ifdef STREAM_COMPACTION
		markPathSegment << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_bools, dev_nbools, dev_paths);
		int lastBool;
		int scanSum;
		cudaMemcpy(&lastBool, dev_bools + pixelcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
		
		thrust::device_ptr<int> dv_in(dev_bools);
		thrust::device_ptr<int> dv_out(dev_scanBools);
		thrust::exclusive_scan(dv_in, dv_in + pixelcount, dv_out);
		cudaMemcpy(&scanSum, dev_scanBools + pixelcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
		scanSum += lastBool;
		

		if (scanSum > 0) {
			thrust::device_ptr<int> dv_nin(dev_nbools);
			thrust::device_ptr<int> dv_nout(dev_scanNBools);
			thrust::exclusive_scan(dv_nin, dv_nin + pixelcount, dv_nout);
			pathSegmentScatter << <numBlocksPixels, blockSize1d >> > (pixelcount, scanSum, dev_tempPaths, dev_paths, dev_bools, dev_scanBools, dev_scanNBools);
			PathSegment* temp = dev_tempPaths;
			dev_tempPaths = dev_paths;
			dev_paths = temp;
		}

#ifdef DEBUG_OUTPUT
		std::cout << "iter-" << iter << ", depth-" << depth << ", paths: " << num_paths << " -> " << scanSum << std::endl;
#endif

		num_paths = scanSum;
#endif

		iterationComplete = num_paths <= 0 || depth >= traceDepth;
	
		if (guiData != NULL){ guiData->TracedDepth = depth; }
	}

	checkCUDAError("pathtrace before finalGather");

	// Assemble this iteration and apply it to the image
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	checkCUDAError("pathtrace finalGather");

	/////////////////////////////////////////////////////////////////////////////

#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
	cudaEventElapsedTime(&elapsed_time_gpu_ms, event_start, event_end);
	cout << "iter-" << iter << ", pathTrace time = " << elapsed_time_gpu_ms << std::endl;
#endif

	retrieveImage();
}

void retrieveImage() {
	// Retrieve image from GPU
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace final cudaMemcpy");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBufferDepth(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// std::cout << "showGBufferDepth" << std::endl;
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferDepthToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showGBufferPos(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// std::cout << "showGBufferNormal" << std::endl;
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferPosToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam, cam.resolution, dev_gBuffer);
}

void showGBufferNormal(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// std::cout << "showGBufferNormal" << std::endl;
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferNormalToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void showDenoisedImage(uchar4* pbo, int iter, float colWeight, float norWeight, float posWeight, int filterSize) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	cudaEventRecord(event_start);
#endif

	for (int i = 0; (1 << i) <= filterSize; i++) {
		denoise << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, cam,
			colWeight, norWeight, posWeight, 1 << i,
			iter, dev_image, dev_gBuffer);
	}

#ifdef RECORD_PATH_TRACE_AND_DENOISE_TIME
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
	cudaEventElapsedTime(&elapsed_time_gpu_ms, event_start, event_end);
	cout << "denoise time = " << elapsed_time_gpu_ms << std::endl;
#endif
	
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void showGaussianBlurImage(uchar4* pbo, int iter, int filterSize) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	gaussianBlur << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_image, filterSize);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}