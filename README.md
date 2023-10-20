CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Yuanqi Wang
  * [LinkedIn](https://www.linkedin.com/in/yuanqi-wang-414b26106/), [GitHub](https://github.com/plasmas).
* Tested on: Windows 11, i5-11600K @ 3.91GHz 32GB, RTX 4090 24GB (Personal Desktop)

# Overview

Original (10 iters, 8 depth)   |  Denoised (filter size 65x65)
:-------------------------:|:-------------------------:
![Original](./img/original.png)  |  ![Denoised](./img/denoised_65x65.png)

A CUDA-based A-Trous denoiser for path tracers.

In real-time rendering, even modern machines don't have the capability to perform an sufficient number of path tracing iterations to achieve a photo-realistic image. Within a few iterations, the image generated can have a lot of noise, resulting in high variance. Therefore, denoising is need to smooth the image. A naive Gaussian blur can indeed smooth the image, but will blur across edges. Also, using a large Gaussian kernel is expensive when performing convolution. Therefore, we use the A-Trous wavelet filter with an edge-stopping function.

This denoiser implements the method by [Dammertz et al.](https://jo.dreggn.org/home/2010_atrous.pdf), with a few tweaks.

# Performance Analysis

## Denoising Overhead

The time cost of denoising is dependent on how many iterations are performed. We record the time cost by computing the denoised image, at different number of iterations requested.

To unify the benchmarks, without explicit mention, all tests are performed using [Cornell Ceiling Light](scenes\cornell_ceiling_light.txt) scene, at the default resolution and camera position. A total of 10 path trace iterations are performed, with a depth of 8 each. Denoising parameters are default as `color_phi = 0.450`, `normal_phi = 0.350`, `position_phi = 0.200`. Image resolution is by default `800x800`

