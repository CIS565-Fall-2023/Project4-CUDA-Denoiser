#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define STACKSIZE 8192

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
        /*float timeToIntersect = gBuffer[index].t * 256.0;

        pbo[index].w = 0;
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;*/
        /*glm::vec3 normal = glm::normalize(gBuffer[index].normal);

        pbo[index].w = 0;
        pbo[index].x = glm::abs(normal.x)* 255.0f;
        pbo[index].y = glm::abs(normal.y)* 255.0f;
        pbo[index].z = glm::abs(normal.z)* 255.0f;*/

        glm::vec3 position = gBuffer[index].position/10.0f;
        if(gBuffer[index].t==-1)
          position=glm::vec3(0.0f);
        pbo[index].w = 0;
        pbo[index].x = glm::abs(position.x)* 255.0f;
        pbo[index].y = glm::abs(position.y)* 255.0f;
        pbo[index].z = glm::abs(position.z)* 255.0f;
    }
}

__global__ void KernelGenerate(int gridLength, float* k1D, float* kernel, glm::ivec2* offset)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(x<gridLength&&y<gridLength){
    kernel[y*gridLength+x]=k1D[x]*k1D[y];
    offset[y*gridLength+x]=glm::ivec2(x-gridLength/2,y-gridLength/2);
  }
}
//ui_filterSize,Gkernel,phi,Goffset
__global__ void generateGuassianKernel(int gridLength, float* kernel, float phi, glm::ivec2* offset)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(x<gridLength&&y<gridLength){
    float center=gridLength/2;
    kernel[y*gridLength+x]=1/(2*glm::pi<float>()*phi*phi)*glm::exp(-((float)((x-center)*(x-center)+(y-center)*(y-center)))/(2*phi*phi));
    offset[y*gridLength+x]=glm::ivec2(x-gridLength/2,y-gridLength/2);
  }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static glm::vec3 * dev_denoise_image = NULL;
static glm::vec3 * dev_denoise_imageC = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

static GBufferPixel* dev_gBuffer = NULL;
static float kernelHost[5]={1.0f/16.0f,1.0f/4.0f,3.0f/8.0f,1.0f/4.0f,1.0f/16.0f};
static float* kernel;
static glm::ivec2* offset;
static float* kernel1D;
static float* Gkernel;
static glm::ivec2* Goffset;

static int *dev_keys;
static int *dev_values;
static PathSegment* dev_pathR;
static ShadeableIntersection* dev_intersectionsR;
static PathSegment* finalbuffer;
static ShadeableIntersection* firstBounce = NULL;
static PathSegment* firstBounceP=NULL;
static BVHnode* dev_tree=NULL;
static glm::vec3* textimgpixel=NULL;
static int* dev_lights=NULL;
static float* dev_lights_area=NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
__global__ void getImage(glm::ivec2 resolution, glm::vec3* image, int iter){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    image[index]/=(float)iter;
  }
}

__global__ void KernelConvolve(GBufferPixel* gBuffer,glm::ivec2 resolution,
  glm::vec3* image, glm::vec3* den_image,float* kernel,glm::ivec2* offset,int length,float c_phi,float n_phi,float p_phi,int stepwidth,int max_step) {
 int x = (blockIdx.x * blockDim.x) + threadIdx.x;
 int y = (blockIdx.y * blockDim.y) + threadIdx.y;

 if (x < resolution.x && y < resolution.y) {
   int index = x + (y * resolution.x);
   
   if(gBuffer[index].t==-1){
     den_image[index]= image[index];
     return;
   }

   glm::ivec2 cuv=glm::ivec2(x,y);
   glm::vec3 sum  = glm::vec3(0.0f);
   glm::vec3 cval = image[index];
   glm::vec3 nval = gBuffer[index].normal;
   glm::vec3 pval = gBuffer[index].position;
   float cum_w = 0.0f;
   for(int i = 0; i < length*length; i++) {
     glm::ivec2 uv = cuv + offset[i]*stepwidth;
     if(offset[i].x*stepwidth>max_step/2||offset[i].y*stepwidth>max_step/2||offset[i].x*stepwidth<-max_step/2||offset[i].y*stepwidth<-max_step/2){
      continue;
     }
     uv.x=glm::clamp(uv.x,0,resolution.x);
     uv.y=glm::clamp(uv.y,0,resolution.y);
     
     int newIdx=uv.x + (uv.y * resolution.x);
     if(gBuffer[newIdx].t==-1){
       continue;
     }
     glm::vec3 ctmp = image[newIdx];
     glm::vec3 t = cval - ctmp;
     float dist2 = dot(t,t);
     float c_w = glm::min((float)glm::exp(-(dist2)/c_phi), 1.0f);
     glm::vec3  ntmp = gBuffer[newIdx].normal;
     t = nval - ntmp;
     dist2 = glm::max(dot(t,t)/(stepwidth*stepwidth),0.0f);
     float n_w = glm::min((float)glm::exp(-(dist2)/n_phi), 1.0f);
     glm::vec3 ptmp = gBuffer[newIdx].position;
     t = pval - ptmp;
     dist2 = dot(t,t);
     float p_w = glm::min((float)glm::exp(-(dist2)/p_phi),1.0f);
     float weight = c_w * n_w * p_w;
     sum += ctmp * weight * kernel[i];
     cum_w += weight*kernel[i];
     /*sum += ctmp * kernel[i];
     cum_w += kernel[i];*/
   }
   den_image[index]=sum/cum_w;
 }
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoise_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoise_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoise_imageC, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoise_imageC, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tree, scene->BVH.size() * sizeof(BVHnode));
	  cudaMemcpy(dev_tree, scene->BVH.data(), scene->BVH.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&Gkernel, pixelcount * sizeof(float));
    cudaMalloc(&Goffset, pixelcount * sizeof(glm::vec2));

    cudaMalloc(&textimgpixel, scene->imgtext.size() * sizeof(glm::vec3));
	  cudaMemcpy(textimgpixel, scene->imgtext.data(), scene->imgtext.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    
    cudaMalloc(&kernel1D,5* sizeof(float));
    cudaMemcpy(kernel1D,kernelHost,5 * sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&kernel, 25 * sizeof(float));
    cudaMalloc(&offset, 25 * sizeof(glm::ivec2));
    // TODO: initialize any extra device memeory you need
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2dKernel(
      (5 + blockSize2d.x - 1) / blockSize2d.x,
      (5 + blockSize2d.y - 1) / blockSize2d.y);

    KernelGenerate<<<blocksPerGrid2dKernel,blockSize2d>>>(5,kernel1D,kernel,offset);

    cudaMalloc(&dev_lights, scene->Lights.size() * sizeof(int));
    cudaMemcpy(dev_lights,  scene->Lights.data(), scene->Lights.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_lights_area, scene->LightArea.size() * sizeof(int));
    cudaMemcpy(dev_lights_area,  scene->LightArea.data(), scene->LightArea.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dev_keys, pixelcount * sizeof(int));
    cudaMalloc((void**)&dev_values, pixelcount * sizeof(int));
    cudaMalloc((void**)&finalbuffer, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_intersectionsR, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&firstBounce, pixelcount  * sizeof(ShadeableIntersection));
    cudaMalloc(&firstBounceP, pixelcount  * sizeof(PathSegment));
    cudaMalloc(&dev_pathR, pixelcount * sizeof(PathSegment));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(kernel);
    cudaFree(offset);
    cudaFree(kernel1D);
    cudaFree(dev_values);
    cudaFree(dev_pathR);
    cudaFree(dev_intersectionsR);
    cudaFree(firstBounce);
    cudaFree(firstBounceP);
    cudaFree(dev_keys);
    cudaFree(finalbuffer);
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
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
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
		glm::vec2 uv=glm::vec2(0.0f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		bool temp_outside=true;
		glm::vec3 temp_dpdu=glm::vec3(0.0f);
		glm::vec3 temp_dpdv=glm::vec3(0.0f);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv=glm::vec2(0.0f);
		glm::vec3 dpdu=glm::vec3(0.0f);
		glm::vec3 dpdv=glm::vec3(0.0f);

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
			}else if (geom.type == TRIANGLE){
				t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_uv,temp_dpdu, temp_dpdv, tmp_normal, temp_outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv=tmp_uv;
				outside=temp_outside;
				dpdu=temp_dpdu;
				dpdv=temp_dpdv;
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
			intersections[path_index].outside=outside;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].dpdu = dpdu;
			intersections[path_index].dpdv = dpdv;
		}
	}
}

__global__ void computeIntersectionsBVH(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, BVHnode* tree
	, int tree_size
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
		glm::vec2 uv = glm::vec2(0.0f);
		glm::vec3 dpdu=glm::vec3(0.0f);
		glm::vec3 dpdv=glm::vec3(0.0f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		bool temp_outside=true;
		int stack[STACKSIZE];
		int stackptr=0;
		int stacksize=0;
		stack[0]=tree_size-1;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv= glm::vec2(0.0f);
		glm::vec3 temp_dpdu=glm::vec3(0.0f);
		glm::vec3 temp_dpdv=glm::vec3(0.0f);

		// naive parse through global geoms
		while(true){
			BVHnode& node=tree[stack[stackptr]];
			stackptr=(stackptr+1)%STACKSIZE;;
			if(node.leaf){
				Geom& geom = geoms[node.geom];
				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				}else if (geom.type == TRIANGLE){
					t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect,tmp_uv,temp_dpdu, temp_dpdv,tmp_normal, temp_outside);
				}
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = geom.materialid;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
					uv= tmp_uv;
					dpdu=temp_dpdu;
					dpdv=temp_dpdv;
					outside=temp_outside;
				}
			}else{
				float dist1=aabbIntersectionTest(tree[node.leftchild], pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				if(dist1>=0.0f){
					stacksize=(stacksize+1)%STACKSIZE;;
					stack[stacksize]=node.leftchild;
				}
				if(node.rightchild!=-1){
					float dist2=aabbIntersectionTest(tree[node.rightchild], pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
					if(dist2>=0.0f){
						stacksize=(stacksize+1)%STACKSIZE;
						stack[stacksize]=node.rightchild;
					}
				}
			}
			if(stackptr==stacksize+1){
				break;
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
			intersections[path_index].outside=outside;
			intersections[path_index].materialId = hit_geom_index;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].dpdu = dpdu;
			intersections[path_index].dpdv = dpdv;
		}
	}
}

__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, glm::vec3 back
	, glm::vec3* textPixel
	, Geom* geoms
	, int* Lights
	, float* LightArea
	, int lightsize
	, int shading
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
			int index=((int)(u01(rng)*((float)lightsize/2)));
			int start=Lights[index*2];
			int end=Lights[index*2+1];
			int gidx=u01(rng)*(end-start)+start;
			if (intersection.materialId == 5)
				int test = 1;
			scatterRay(pathSegments[idx],intersection,materials[intersection.materialId],rng,textPixel,back,geoms[gidx],LightArea[index],shading);
			// If the material indicates that the object was a light, "light" the ray
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color *= back;
			pathSegments[idx].remainingBounces=0;
		}
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
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].normal=shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].position=getPointOnRay(pathSegments[idx].ray,gBuffer[idx].t);
    /*
    glm::vec3 normal=shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].normal=glm::vec3(glm::abs(normal.x),glm::abs(normal.y),glm::abs(normal.z));
    glm::vec3 position=getPointOnRay(pathSegments[idx].ray,gBuffer[idx].t)/10.0f;
    gBuffer[idx].position=glm::vec3(glm::abs(position.x),glm::abs(position.y),glm::abs(position.z));
    */
    //atomicAdd(c_phi, float val);
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void setfinalbuffer(int nPaths, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		iterationPaths[index].pixelIndex=index;
		iterationPaths[index].color=glm::vec3(0.0f);

	}
}

__global__ void materialRemap(int num_paths,
	PathSegment* dev_paths,
	ShadeableIntersection* intersection,
	int *dev_keys,
	int *dev_values){

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_paths)
	{
		dev_keys[index]=intersection[index].materialId;
		dev_values[index]=index;
	}
}


struct is_zero
{
  __host__ __device__
  bool operator()(PathSegment x)
  {
    return x.remainingBounces  == 0;
  }
};

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
 __global__ void kernScatter(int n, PathSegment* idata,PathSegment* finaldata) {
	// TODO
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	if (idata[index].remainingBounces == 0) {
		finaldata[idata[index].pixelIndex].color=idata[index].color;
	}

}


__global__ void kernReshuffle(int N, int* particleArrayIndices, PathSegment* pos,
	PathSegment* posR, ShadeableIntersection* vel, ShadeableIntersection* velR){
	  int index = threadIdx.x + (blockIdx.x * blockDim.x);
	  if (index >= N) {
		return;
	  }
	  posR[index]=pos[particleArrayIndices[index]];
	  velR[index]=vel[particleArrayIndices[index]];
  
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

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
	const int blockSize1d = 128;
  if(iter==1)
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter,traceDepth, dev_paths);
  else
    cudaMemcpy(dev_paths,firstBounceP,pixelcount*sizeof(PathSegment),cudaMemcpyDeviceToDevice);
  checkCUDAError("generate camera ray");
	
	
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	setfinalbuffer<<<numBlocksPixels, blockSize1d>>>(num_paths,finalbuffer);
	//thrust::device_vector<int> dev_thrust_out(pixelcount);
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if(depth==0&&iter>1){
			cudaMemcpy(dev_intersections,firstBounce,num_paths*sizeof(ShadeableIntersection),cudaMemcpyDeviceToDevice);
			depth++;
		}else{
      computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
        iter
        , num_paths
        , dev_paths
        , dev_geoms
        , dev_tree
        , hst_scene->BVH.size()
        , dev_intersections
      );
		
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			depth++;
			
      materialRemap << <numblocksPathSegmentTracing, blockSize1d >> >(
        num_paths,
        dev_paths,
        dev_intersections,
        dev_keys,
        dev_values
      );
      thrust::sort_by_key(thrust::device,dev_keys, dev_keys+num_paths, dev_values);
      kernReshuffle<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths,dev_values,dev_paths,dev_pathR,dev_intersections,dev_intersectionsR);

      PathSegment *tempdev_paths=dev_pathR;
      dev_pathR=dev_paths;
      dev_paths=tempdev_paths;
      ShadeableIntersection* tempdev_intersections=dev_intersectionsR;
      dev_intersectionsR=dev_intersections;
      dev_intersections=tempdev_intersections;
		}

		if(depth==1&&iter==1){
			cudaMemcpy(firstBounce,dev_intersections,num_paths*sizeof(ShadeableIntersection),cudaMemcpyDeviceToDevice);
			cudaMemcpy(firstBounceP,dev_paths,num_paths*sizeof(PathSegment),cudaMemcpyDeviceToDevice);
		}

		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			hst_scene->backColor,
			textimgpixel,
			dev_geoms,
			dev_lights,
			dev_lights_area,
			hst_scene->Lights.size(),
			0
		);

		kernScatter << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_paths,
			finalbuffer
		);
		
		PathSegment* new_end=thrust::remove_if(thrust::device,dev_paths,dev_paths+num_paths,is_zero());
		
		num_paths=new_end-dev_paths;

		if(depth>traceDepth || num_paths==0){
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
	
	}

		
	
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, finalbuffer);
	
	
	// Assemble this iteration and apply it to the image
	

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering

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

void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

void showDenoisImage(uchar4 *pbo,int ui_filterSize, float ui_colorWeight,float ui_normalWeight,float ui_positionWeight,int iter){
  const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
  
  int filter_iter=ceil(log2(((float)ui_filterSize)/4.0f));
  //int filter_iter=1;
  const int pixelcount = cam.resolution.x * cam.resolution.y;
  cudaMemcpy(dev_denoise_image,dev_image,pixelcount*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
  int stepwidth=1;
  getImage<<<blocksPerGrid2d,blockSize2d>>>( cam.resolution, dev_denoise_image,iter);

  for(int i=0;i<filter_iter;i++){
    KernelConvolve<<<blocksPerGrid2d,blockSize2d>>>(dev_gBuffer, cam.resolution, dev_denoise_image,dev_denoise_imageC,kernel,offset,5,ui_colorWeight,ui_normalWeight,ui_positionWeight,stepwidth,ui_filterSize);
    stepwidth*=2;
    ui_colorWeight = ui_colorWeight / 2;
    glm::vec3* temp=dev_denoise_image;
    dev_denoise_image=dev_denoise_imageC;
    dev_denoise_imageC=temp;
  }
 
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, dev_denoise_image);
  cudaMemcpy(hst_scene->state.image.data(), dev_denoise_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void showGDenoisImage(uchar4 *pbo,int ui_filterSize, float phi,float ui_colorWeight,float ui_normalWeight,float ui_positionWeight,int iter){
  const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    const dim3 blocksPerGrid2dKernel(
      (ui_filterSize + blockSize2d.x - 1) / blockSize2d.x,
      (ui_filterSize + blockSize2d.y - 1) / blockSize2d.y);
	const int pixelcount = cam.resolution.x * cam.resolution.y;
  
  generateGuassianKernel<<<blocksPerGrid2dKernel,blockSize2d>>>(ui_filterSize,Gkernel,phi,Goffset);
  KernelConvolve<<<blocksPerGrid2d,blockSize2d>>>(dev_gBuffer, cam.resolution, dev_image,dev_denoise_image,Gkernel,offset,ui_filterSize,ui_colorWeight,ui_normalWeight,ui_positionWeight,1,ui_filterSize);
 
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, dev_denoise_image);
  cudaMemcpy(hst_scene->state.image.data(), dev_denoise_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
