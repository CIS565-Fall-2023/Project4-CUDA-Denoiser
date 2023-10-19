CUDA Denoiser For CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tianyi Xiao
  * [LinkedIn](https://www.linkedin.com/in/tianyi-xiao-20268524a/), [personal website](https://jackxty.github.io/), [Github](https://github.com/JackXTY).
* Tested on: Windows 11, i9-12900H @ 2.50GHz 16GB, Nvidia Geforce RTX 3070 Ti 8032MB (Personal Laptop)

### Denoised Render Result
| original | denoised |
| :----: | :----: |
| ![](/img/denoise/result_denoise_0_ref.png) | ![](/img/denoise/result_denoise_0.png) |

![](/img/denoise/result_denoise_0.png)

![](/img/denoise/result_denoise_1.png)

### Analysis
By defualt: filter size = 100, resolution = 800 x 800, iteration time = 5, colorWeight = 0.45, normalWeight = 0.35, positionWeight = 0.2.

#### Iteration and Time Analysis
![](/img/denoise/denoise_time_graph.png)

The above graph shows how render change when iteration increase, under same resolution (800x800), filter size(100) and also other parameters. And in my implementation, I only blur the image after last iteration is done, to get the correct final result while saving time in bluring. As we can see, the time added due to denoising is about 10ms, which is basically the time to take one path-trace iteration. This is a reasonable and acceptable time for such kind of post-process.

Denoise with 3 path-trace iteration:
![](/img/denoise/denoise_3.png)
Denoise with 5 path-trace iteration:
![](/img/denoise/denoise_5.png)
Denoise with 10 path-trace iteration:
![](/img/denoise/denoise_10.png)
(Other parameters are not changed.)

From above results we can see that, actually, with 3 path-trace iterations, we can get reasonably acceptable render result. However, the 5 iterations result is better with close looking. So in the following tests, I would use 5 iterations for test.

#### Resolution Analysis
400 x 400:
![](/img/denoise/400.png)
800 x 800:
![](/img/denoise/denoise_5.png)
1600 x 1600:
![](/img/denoise/1600.png)

![](/img/denoise/resolution_time_graph.png)

As we can see, the denoise time increase as resolution increase, since there are more pixels to handle.

And the visual effect of higher resolution pictures is better. I think, generally the high resolution origianl path trace contain more information for denoiser, so it would have better visual effect after denoise. While denoising, for low resolution image, too much details are skipped with fewer pixels.

#### FilterSize Analysis
| 0 | 1 | 2 | 4 |
| :----: | :----: | :----: | :----: |
| ![](/img/denoise/5_0.png) | ![](/img/denoise/5_1.png) | ![](/img/denoise/5_2.png) |  ![](/img/denoise/5_4.png) |


|  8  |  16  |  32  | 64 |
| :----: | :----: | :----: | :----: |
| ![](/img/denoise/5_8.png) | ![](/img/denoise/5_16.png) | ![](/img/denoise/5_32.png)  |  ![](/img/denoise/denoise_5.png) |

While the filter size increase, actuallly the denosie loop times increase. In this denoise algorithm, in i-th loop the kernel size will increase to 2^(i-1). And when the kernel size  will exceed the given filter size, the denosie loop will stop.

In this way, without any doubt, the result is smoother and smoother as filter size increase. And the denoise time is proportional to the loop times.

#### Denoise with Different Material types

![](/img/denoise/result_denoise_1.png)

From this picture, we can see that the flat mirror part is appearently too blur. Because in this algorithm, we decide blur ratio due to color, position and normal difference, and for the flat mirror, there is no difference in normal and only very slight difference in position, so it's only blured according to the color difference, which could make the result inaccurate. What's more, therefore, I think the blur effect for flat texture material would also be bad.

#### Denoise with Different Scenes

![](/img/denoise/result_denoise_0.png)

With tests, I found that, for complicated scenes, we need more path trace iterations to produce acceptable denoised effect. The reason is that, there would be less light hit the light resource within same iteration time.

#### Compared with Gaussian Blur
Gaussian Blur, Kernel Size = 2:
![](/img/denoise/gaussianBlur_2.png)
Gaussian Blur, Kernel Size = 3:
![](/img/denoise/gaussianBlur_3.png)

Appearently, the gaussian blur just blur the whole image without considering any geometry information. Actually the noise points are still there, they just become also less clear with the image together. However, with Edge-Avoiding A-torus, most noise point could be detected and discarded. And also gaussian blur will blur two triangles in different planes or different space (like one behind another), which this denoiser could avoid with position and normal difference.