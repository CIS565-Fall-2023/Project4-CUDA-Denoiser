#pragma once

#include "sceneStructs.h"
#include "utilities.h"
#include "glm/glm.hpp"

// #define GUASSIAN_KERNEL


__constant__ float dev_kernel[25];
__constant__ int dev_offsets[50];

static glm::vec3* dev_denoised_curr_image = NULL;
static glm::vec3* dev_denoised_next_image = NULL;

void initDenoiser() {
#ifdef GUASSIAN_KERNEL
    float host_kernel[25] = {
       0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
       0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
       0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
       0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
       0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f
    };
#else 
    float host_kernel[25] = {
        0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f,
        0.0156f, 0.0625f, 0.0938f, 0.0625f, 0.0156f,
        0.0234f, 0.0938f, 0.1406f, 0.0938f, 0.0234f,
        0.0156f, 0.0625f, 0.0938f, 0.0625f, 0.0156f,
        0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f
    };
#endif

    glm::ivec2 host_offsets[25] = {
        glm::ivec2(-2, -2), glm::ivec2(-1, -2), glm::ivec2(0, -2), glm::ivec2(1, -2), glm::ivec2(2, -2),
        glm::ivec2(-2, -1), glm::ivec2(-1, -1), glm::ivec2(0, -1), glm::ivec2(1, -1), glm::ivec2(2, -1),
        glm::ivec2(-2, 0), glm::ivec2(-1, 0), glm::ivec2(0, 0), glm::ivec2(1, 0), glm::ivec2(2, 0),
        glm::ivec2(-2, 1), glm::ivec2(-1, 1), glm::ivec2(0, 1), glm::ivec2(1, 1), glm::ivec2(2, 1),
        glm::ivec2(-2, 2), glm::ivec2(-1, 2), glm::ivec2(0, 2), glm::ivec2(1, 2), glm::ivec2(2, 2)
    };

    cudaMemcpyToSymbol(dev_kernel, host_kernel, sizeof(host_kernel));
    cudaMemcpyToSymbol(dev_offsets, host_offsets, sizeof(host_offsets));

}

void denoiserFree() {
    cudaFree(dev_denoised_curr_image);
    cudaFree(dev_denoised_next_image);
}

/*
* Get the world intersection point position
*/
__device__ glm::vec3 getWorldPos(Ray r, float t) {
    return r.origin + t * r.direction;
}

__device__ float caculate_weight(float dist, float phi) {
    return glm::min(glm::exp(-(dist) / (phi)), 1.0f);
}


// Computer Wavelet Transformation
__global__ void denoiserKernel(glm::vec3* c_in, glm::vec3* c_out, GBufferPixel* gBuffer,
    glm::ivec2 resolution, size_t stepwidth, float c_phi, float n_phi, float p_phi, int iter) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int currIdx = x + (y * resolution.x);

        glm::vec3 sum = glm::vec3(0.f);
        glm::vec3 cval = c_in[currIdx];
        glm::vec3 nval = gBuffer[currIdx].normal;
        glm::vec3 pval = gBuffer[currIdx].pos;

        float cum_w = 0.0f;

        for (int i = 0; i < 25; ++i) {
            int otherX = x + dev_offsets[i * 2]  * stepwidth;
            int otherY = y + dev_offsets[i * 2 + 1] * stepwidth;

            otherX = glm::clamp(otherX, 0, resolution.x - 1);
            otherY = glm::clamp(otherY, 0, resolution.y - 1);

            int otherIdx = otherX + (otherY * resolution.y);

            glm::vec3 ctmp = c_in[otherIdx];
            glm::vec3 t = cval - ctmp;
            float dist2 = glm::dot(t, t);
            float c_w = caculate_weight(dist2, c_phi);

            glm::vec3 ntmp = gBuffer[otherIdx].normal;
            t = nval - ntmp;
            dist2 = glm::max(glm::dot(t, t) / (stepwidth * stepwidth), 0.0f);
            float n_w = caculate_weight(dist2, n_phi);

            glm::vec3 ptmp = gBuffer[otherIdx].pos;
            t = pval - ptmp;
            dist2 = glm::dot(t, t);
            float p_w = caculate_weight(dist2, p_phi);

            float weight = c_w * n_w * p_w;
            sum += ctmp * weight * dev_kernel[i];
            cum_w += weight * dev_kernel[i];
        }

        c_out[currIdx] = sum / cum_w;
    }
}
