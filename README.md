CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Zhenzhong Tang
  * [LinkedIn](https://www.linkedin.com/in/zhenzhong-anthony-tang-82334a210), [Instagram](https://instagram.com/toytag12), [personal website](https://toytag.net/)
* Tested on: Windows 11 Pro 22H2, AMD EPYC 7V12 64-Core Processor (4 vCPU cores) @ 2.44GHz 28GiB, Tesla T4 16GiB (Azure)

## Overview

A path tracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter.

|Cornell|Stanford Bunny|
|:-:|:-:|
|![](img/cornell_cmp.JPEG)|![](img/bunny_cmp.JPEG)|

The denoise filter is based on the paper ["Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch](https://jo.dreggn.org/home/2010_atrous.pdf).

### Filter

The filter is based on B3-spline interpolation $(1/16, 1/4, 3/8, 1/4, 1/16)$ and extended to a $5\times5$ kernel. The kernel is applied in a-trous style, iteratively increasing the step width by a factor of 2.

![](img/a-trous.png)

However the filter itself is not edge-aware, meaning that it will smooth across edges and act essentially as a Gaussian blur.

| 80x80 A-Trous Filter | 80x80 Gaussian Filter |
|:--:|:--:|
|![](img/cornell_blur.png)|![](img/gaussian_blur.png)|

### G-buffer

The G-buffer contains the following information:
- Normal
- Position

| Normal | Position |
|:--:|:--:|
|![](img/cornell_normal.png)|![](img/cornell_position.png)|
|![](img/bunny_normal.png)|![](img/bunny_position.png)|

### Results

Combined with G-Buffer, the filter becomes edge-aware, meaning that it will not smooth across edges. The result is a much cleaner image.

| Original | 80x80 Filter | Reference (1000 iterations) |
|:--:|:--:|:--:|
|![](img/cornell_original.png)|![](img/cornell_80x80.png)|![](img/cornell_reference.png)|

Default Cornell scene with 10 samples per pixel, 800x800 resolution, 8 light bounces per sample.

| Original | 80x80 Filter | Reference (1000 iterations) |
|:--:|:--:|:--:|
|![](img/bunny_original.png)|![](img/bunny_80x80.png)|![](img/bunny_reference.png)|

Slightly more complex Stanford Bunny scene with 10 samples per pixel, 800x800 resolution, 8 light bounces per sample.

## Performance Analysis

We have implemented the denoiser in CUDA, and applied some memory footprint optimizations to our G-buffer.
- We adopted oct-encoding for normals, which reduces the `vec3` normal to `vec2`.
- Instead of storing the full `vec3` position, we only store a `float` z-depth and use the pixel index to reconstruct the world space position.

The following graph compares the influence of filter size and image resolution on our denoising filters. It is tested on the stanford bunny scene with 10 samples per pixel, 800x800 resolution, 8 light bounces per sample.

![](img/filter_size_cmp.svg)

From the comparison above, we can see that as the filter size doubles, the average runtime increases by approximately the runtime of one iteration of the filter. This is expected, as the a-trous filter is applied iteratively, and the runtime of each iteration is approximately the same. We can see the trend lines are also linear.

Our filter is heavily memory bound, meaning that by limiting the memory footprint of G-buffer, there are significant reduction in global memory access, and resulting in huge performance gain.

The next graph compares the influence of image resolution on our denoising filters. It is also tested on the stanford bunny scene with 10 samples per pixel, 80x80 filter size, 8 light bounces per sample.

![](img/resolution_cmp.svg)

There runtime is proportional to the number of pixels in the image. Since the resolution is increasing exponentially, the trend line is also exponential. With more heavy global memory access, G-buffer optimization is more effective.

We also have detail results for the filters with different sizes in different scenes. Shown below.

| Original | 5x5 Filter | 10x10 Filter |
|:--:|:--:|:--:|
|![](img/cornell_original.png)|![](img/cornell_5x5.png)|![](img/cornell_10x10.png)|
| **20x20 Filter** | **40x40 Filter** | **80x80 Filter** |
|![](img/cornell_20x20.png)|![](img/cornell_40x40.png)|![](img/cornell_80x80.png)|

| Original | 5x5 Filter | 10x10 Filter |
|:--:|:--:|:--:|
|![](img/bunny_original.png)|![](img/bunny_5x5.png)|![](img/bunny_10x10.png)|
| **20x20 Filter** | **40x40 Filter** | **80x80 Filter** |
|![](img/bunny_20x20.png)|![](img/bunny_40x40.png)|![](img/bunny_80x80.png)|

All scenes are rendered with 10 samples per pixel, 800x800 resolution, 8 light bounces per sample.

Across different scenes, cornell and bunny are two of the most representative scenes. The cornell scene is simple and mostly diffusive materials. Our denoising filter performed very well in this case. On the other hand, the bunny scene is more complex, with more specular and refractive materials. We can clearly see that the filter is starting to blur some details of the bunny and the reflection and refraction is not nearly as clear as the reference image. However, the filter performed consistently in different lighting conditions, in most cases there are only minor artifacts/over-exposure.

This is generally a good results, though it might not look as good in merely 10 iterations. The denoising filter significantly reduces the number of iterations needed to achieve a decent result. Typically 100 iterations with denoising is comparable to or even better than 1000 iterations without denoising.
