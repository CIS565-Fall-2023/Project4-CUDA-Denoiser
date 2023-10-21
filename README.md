CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

Han Wang

Tested on: Windows 11, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz 22GB, GTX 3070 Laptop GPU


|without Denoize|with Denoize|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-19_01-52-06z.181samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-20_23-54-52z.277samp.png" width="300" height="300">
### Analysis
Firstly, Among the tested output, since the denoize time will be affected by the various variables, I analyzed the denoize in the default condition given to us:
800x800 resolution, 80 filter size. The output is: 4.645Ms


Based on my observation, with the denoize feature implemented, the iteration amount needed for reaching the acceptable smooth condition was significantly reduced, leading to more efficient and time-saving results in the process.

Also, based on my test I got the following graph amount of the resolution and runtime of denoize:

<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/denoize_resolution.png width="600" height="600"

It is easy to see that while the resolution increased, the amount of that denoize needed was increased. It is also easy to understand: there are more pixels needed to estimate, and thus takes more time to denoize.

