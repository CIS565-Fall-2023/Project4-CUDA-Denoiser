#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define PIXEL_NUM 25

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
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
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].normal.x * 256.0f;

        pbo[index].w = 0;
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
__constant__ float dev_filter[25];
__constant__ float2 dev_offset[25];
__constant__ float dev_offset_x[25];
__constant__ float dev_offset_y[25];
static glm::vec3* dev_denoise_1 = NULL;
static glm::vec3* dev_denoise_2 = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    float filter[25] = {
      1.f/256.f, 4.f/256.f, 6.f/256.f, 4.f/256.f, 1.f/256.f,
      4.f/256.f, 16.f/256.f, 24.f/256.f, 16.f/256.f, 4.f/256.f,
      6.f/256.f, 24.f/256.f, 36.f/256.f, 24.f/256.f, 6.f/256.f,
      4.f/256.f, 16.f/256.f, 24.f/256.f, 16.f/256.f, 4.f/256.f,
      1.f/256.f, 4.f/256.f, 6.f/256.f, 4.f/256.f, 1.f/256.f
    };

    float2 offset[25] = {
      make_float2(-2, -2), make_float2(-2, -1), make_float2(-2, 0), make_float2(-2, 1), make_float2(-2, 2),
      make_float2(-1, -2), make_float2(-1, -1), make_float2(-1, 0), make_float2(-1, 1), make_float2(-1, 2),
      make_float2(0, -2), make_float2(0, -1), make_float2(0, 0), make_float2(0, 1), make_float2(0, 2),
      make_float2(1, -2), make_float2(1, -1), make_float2(1, 0), make_float2(1, 1), make_float2(1, 2),
      make_float2(2, -2), make_float2(2, -1), make_float2(2, 0), make_float2(2, 1), make_float2(2, 2)
    };

    float offset_x[25] = {
      -2, -1, 0, 1, 2,
      -2, -1, 0, 1, 2,
      -2, -1, 0, 1, 2,
      -2, -1, 0, 1, 2,
      -2, -1, 0, 1, 2
		};

    float offset_y[25] = {
      -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2
    };


    cudaMemcpyToSymbol(dev_filter, filter, 25 * sizeof(float));
    checkCUDAError("pathtraceInit1");

    cudaMemcpyToSymbol(dev_offset, offset, 25 * sizeof(float2));
    checkCUDAError("pathtraceInit2");
    cudaMemcpyToSymbol(dev_offset_x, offset_x, 25 * sizeof(float));
    checkCUDAError("pathtraceInit2");
    cudaMemcpyToSymbol(dev_offset_y, offset_y, 25 * sizeof(float));
    checkCUDAError("pathtraceInit2");
    cudaMalloc(&dev_denoise_1, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceInit3");
    cudaMalloc(&dev_denoise_2, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceInit4");
    cudaMemset(dev_denoise_1, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceInit5");
    cudaMemset(dev_denoise_2, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceInit6");

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    cudaFree(dev_denoise_1);
    cudaFree(dev_denoise_2);


    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
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
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
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
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeSimpleMaterials (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces == 0) {
      return;
    }

    if (intersection.t > 0.0f) { // if the intersection exists...
      segment.remainingBounces--;
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        segment.color *= (materialColor * material.emittance);
        segment.remainingBounces = 0;
      }
      else {
        segment.color *= materialColor;
        glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
        scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      segment.color = glm::vec3(0.0f);
      segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
  }
}

__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,   
  PathSegment* pathSegments,
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_paths)
  {
      ShadeableIntersection intersection = shadeableIntersections[idx];
      PathSegment segment = pathSegments[idx];
      GBufferPixel gbufferPixel = gBuffer[idx];

      //if (intersection.t > 0.0f) { // if the intersection exists...
        gBuffer[idx].normal = intersection.surfaceNormal;
        gBuffer[idx].pos = getPointOnRay(segment.ray, intersection.t);
      //}

      //else
      //{
				//gBuffer[idx].normal = glm::vec3(0.0f);
				//gBuffer[idx].pos = glm::vec3(0.0f);
      //}
  }
}

// Add the current iteration's output to the overall image
  __global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void denoise_a_trois(glm::vec3 * image, float depth,
                                GBufferPixel* gBuffer, 
                                int stepWidth, glm::vec2 resolution, glm::vec3 * image_output,
                                float *offset_x, float *offset_y, float *filter,
                                float color_phi, float normal_phi, float pos_phi)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int resolution_x = resolution.x;
  int resolution_y = resolution.y;
  if (x >= resolution.x || y >= resolution.y)
  {
    return;
  }

  float total_w = 0.0;
  glm::vec3 sum_color = glm::vec3(0);
  glm::vec2 pixel = glm::vec2(x, y);
  glm::vec3 color = image[x + y * resolution_x];
  printf("test3\n");
  GBufferPixel gbufferPixel = gBuffer[x + y * resolution_x];
  printf("test3.1\n");
  glm::vec3 normal = gbufferPixel.normal;
  glm::vec3 pos = gbufferPixel.pos;
  printf("test3.2\n");

  for (int i = 0; i < PIXEL_NUM; i++)
  {
    printf("test4\n");
    glm::vec2 curr_offset = glm::vec2(offset_x[i], offset_y[i]);
    printf("test4.1\n");
    glm::vec2 curr_pixel = pixel + curr_offset * (float) stepWidth;
    printf("test4.2\n");
    glm::clamp(curr_pixel, glm::vec2(0.0), resolution - glm::vec2(1.0));
    printf("test4.3\n");

    int curr_pixel_x = curr_pixel.x;
    int curr_pixel_y = curr_pixel.y;
    glm::vec3 curr_color = image[curr_pixel_x + curr_pixel_y * resolution_x];
    curr_color /= (float)depth;
    glm::vec3 color_t = color - curr_color;
    float dist_color = glm::dot(color_t, color_t);
    float color_w = glm::min(exp(-(dist_color) / color_phi), 1.0f);
    printf("test5\n");

    glm::vec3 curr_normal = gBuffer[curr_pixel_x + curr_pixel_y * resolution_x].normal;
    glm::vec3 normal_t = normal - curr_normal;
    float dist_normal = glm::dot(normal_t, normal_t);
    float normal_w = glm::min(exp(-(dist_normal) / normal_phi), 1.0f);
    printf("test6\n");

    glm::vec3 curr_pos = gBuffer[curr_pixel_x + curr_pixel_y * resolution_x].pos;
    glm::vec3 pos_t = pos - curr_pos;
    float dist_pos = glm::dot(pos_t, pos_t);  
    float pos_w = glm::min(exp(-(dist_pos) / pos_phi), 1.0f);
    printf("test7\n");

    float weight = color_w * normal_w * pos_w;
    sum_color += color * weight * filter[i];
    total_w += weight * filter[i];
    //printf("test loop\n");
  }
  image_output[x + y * resolution_x] = sum_color / total_w;
  //printf("test5\n");
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Pathtracing Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * NEW: For the first depth, generate geometry buffers (gbuffers)
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally:
    //     * if not denoising, add this iteration's results to the image
    //     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  bool iterationComplete = false;
	while (!iterationComplete) {

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();

  if (depth == 0) {
    generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
  }

	depth++;

  shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  iterationComplete = depth == traceDepth;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);


    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
}

void showDenoise(uchar4* pbo, int iteration, int ui_filterSize, float ui_colorWeight, float ui_normalWeight, float ui_positionWeight)
{
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
  //printf("testdenoise 1\n");
  denoise_a_trois << <blocksPerGrid2d, blockSize2d>> > (dev_image, iteration,
    dev_gBuffer, 1,
    cam.resolution, dev_denoise_1,
    dev_offset_x, dev_offset_y, dev_filter,
    ui_colorWeight, ui_normalWeight, ui_positionWeight);
  checkCUDAError("denoise1");
  //printf("testdenoise 2\n");

  // not sure if < or <=
  for (int i = 1; (1 << i) < (ui_filterSize >> 1); i++)
  {
    int stepWidth = 1 << i;
    denoise_a_trois << <blocksPerGrid2d, blockSize2d >> > (dev_denoise_1, iteration,
      dev_gBuffer, stepWidth,
      cam.resolution, dev_denoise_2,
      dev_offset_x, dev_offset_y, dev_filter,
      ui_colorWeight, ui_normalWeight, ui_positionWeight);
    std::swap(dev_denoise_1, dev_denoise_2);
    //printf("testdenoise loop %d\n", i);
    checkCUDAError("denoise2");

  }

  // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
  //gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
  // maybe no iteration
  //printf("testdenoise 3\n");
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iteration, dev_denoise_1);
  checkCUDAError("denoise3");

}

void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}
