#pragma once

#include "sceneStructs.h"
#include "utilities.h"

/*
* Get the world intersection point position
*/
__host__ __device__ glm::vec3 getWorldPos(Ray r, float t) {
    return r.origin + t * r.direction;
}
//
//__global__ void wavelet_decomposition(glm::vec3* c_in, glm::vec3* c_out,
//    glm::ivec3 resolution, int stride) {
//
//    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//    if (x < resolution.x && y < resolution.y) {
//        int index = x + (y * resolution.x);
//    }
//}
//
//void compute_wavelet_transform(glm::vec3* d_image, glm::vec3* c_image,
//    glm::ivec2 resolution, int pixelcount, int N) {
//
//    glm::vec3* c_current, * c_next;
//
//    cudaMalloc(&c_current, pixelcount * sizeof(glm::vec3));
//    cudaMemcpy(c_current, c_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
//
//    cudaMalloc(&c_next, pixelcount * sizeof(glm::vec3));
//
//    for (int i = 0; i < N; ++i) {
//        int stride = 1 << i;
//
//
//    }
//
//}