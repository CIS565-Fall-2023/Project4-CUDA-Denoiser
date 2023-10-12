# CUDA Denoiser For CUDA Path Tracer

<img src="img/renders/cornell_ceil_regular_100spp.png" width="50%" /><img src="img/renders/cornell_ceil_denoised_filter5_100spp.png" width="50%" />

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Aditya Gupta
  * [Website](http://adityag1.com/), [GitHub](https://github.com/AdityaGupta1), [LinkedIn](https://www.linkedin.com/in/aditya-gupta1/), [3D renders](https://www.instagram.com/sdojhaus/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, NVIDIA GeForce RTX 2070 8GB (personal laptop)
  * Compute capability: 7.5

## Introduction

This project is a CUDA implementation of [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf), a denoising method for path tracers. This method involves using successive blurs of increasing size while weighting based on differences in color, position, and normals. The above images show a comparison of the raw path traced output at 100 spp (left) and the denoised version (right).

## Features

![](img/renders/gui.png)

The GUI has controls for enabling denoising, setting the filter size, changing weight parameters, and outputting the G-buffer.

## Performance Analysis

### Image Resolution and Filter Size vs. Time/Frame

I first compared the path tracer's performance across multiple resolutions and denoising filter sizes. A filter size of 0 means denoising was disabled. A filter size of $k \geq 1$ means the filter ran $k$ times per path tracing iteration, and for each filter iteration $i$ such that $1 \leq i \leq k$, the filter kernel covered $5 \cdot 2^{i - 1}$ pixels in each direction.

![](img/charts/image_res_filter_size.png)

From this graph, we can see that for all tested resolutions, the time per frame increases linearly with the number of filter iterations.

### Filter Size vs. Image Quality

This brings up a question: does an increased filter size necessarily mean better results? Here are some test images:

| Filter Size | Image |
|-------------|-------|
| None        | ![](img/renders/cornell_ceil_regular_100spp.png) |
| 1           | ![](img/renders/cornell_ceil_denoised_filter1_100spp.png) |
| 3           | ![](img/renders/cornell_ceil_denoised_filter3_100spp.png) |
| 5           | ![](img/renders/cornell_ceil_denoised_filter5_100spp.png) |
| 7           | ![](img/renders/cornell_ceil_denoised_filter7_100spp.png) |

There are differences in the first four images (up to and including size 5), but after size 5 the difference is negligible. This is because pixels that are very far away (e.g. $\frac{1}{2} \cdot 5 \cdot 2^{7-1} = 160$ pixels away for filter size 7) have very different positions and likely very different colors and normals, so their weights will be low.

### Total Iterations Needed

TODO: num iterations needed for "acceptably smooth", include image diff

### Material Comparisons

TODO: compare diffuse and specular materials, maybe include image diff or zoomed in comparison
