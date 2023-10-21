CUDA Denoising
================
![Path Tracer](/img/.2023-10-20_16-02-44z.104samp.png)

Jason Xie, Fall 2023

[LinkedIn](https://www.linkedin.com/in/jia-chun-xie/)
[Website](https://www.jchunx.dev)

## Overview

Improving the quality of Monte Carlo path tracing through A-Trous wavelet filtering.

Taking advantage of the fact that nearby pixels are likely to have similar colors, A-trous wavelet filtering is used to smooth out noise in the image. This is done by convolving the image with a B3 spline kernel, which is a weighted average of the pixel's neighbors. The kernel is applied multiple times to the image, with each iteration doubling the size of the kernel while skipping pizels to ensure uniform computational load. In addition to smoothing, the edge stopping function prevents sharp edges from being blurred out by taking color, normals, and positions of the pixels into account.

## Analysis

### 📈 Performance

The effect of various filter sizes and render resolutions on filtering time is shown.

Since the number of denoise iterations is the log of filter size, we can see a linear trend betweent the denoise time and the log of the filter size.

![denoise_times](/img/denoise_times.png)

### ✨ Quality

_**how does it influence the number of iterations required for an acceptable result?**_

| Without Filtering (100 iterations) | With Filtering (100 iterations) | Gaussian Blur |
|-----------------|--------------------|-----------------|
| ![without filtering](/img/.2023-10-20_16-07-06z.101samp.png) | ![with filtering](/img/.2023-10-20_16-02-44z.104samp.png) | ![gaussian blur](/img/imageedit_1_5731991672.png) |

Filtering darmatically reduces the number of iterations required to achieve an acceptable result.

_**how does filter size affect the quality of the result?**_

| Filter Size: 5 | Filter Size: 20 | Filter Size: 200 |
|-----------------|-----------------|-----------------|
| ![filter size 5](/img/cornell.2023-10-20_15-38-38z.357samp.png) | ![filter size 20](/img/cornell.2023-10-20_15-52-55z.247samp.png) | ![filter size 200](/img/cornell.2023-10-20_15-49-40z.179samp.png) |

It looks like increasing the filter size has diminishing effects on the image quality. In addition, images with larger filter sizes appear to be darker. This may be due to the fact that A-Trous filtering is not energy conserving.

_**how effective is it for different materials?**_

| Without Filtering (Specular) | With Filtering (Specular) |
|-----------------|-----------------|
| ![without filtering](/img/cornell.2023-10-20_15-57-50z.198samp.png) | ![with filtering](/img/cornell.2023-10-20_16-00-30z.208samp.png) |

While A-Trous filtering preserves edges of surfaces, it does not preserve edges present in the reflections of objects. This is due to the fact that g-buffers do not take reflections into account.

_**how effective is it for different scenes?**_

| Cornell Box | Cornell Box (Ceiling Light) |
|-----------------|-----------------|
| ![cornell box](/img/cornell.2023-10-21_02-25-25z.118samp.png) | ![cornell box ceiling light](/img/cornell.2023-10-21_02-18-20z.10samp.png) |

I found is more difficult for A-Trous filtering to denoise scenes with low lighting conditions. More more light rays and therefore colored pixels available, the filtering is able to take advantage of the extra information to produce a better picture.