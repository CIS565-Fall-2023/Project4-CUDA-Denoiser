CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

Han Wang

Tested on: Windows 11, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz 22GB, GTX 3070 Laptop GPU

|without denoizer|with denoize|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-19_01-52-06z.181samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-20_23-54-52z.277samp.png" width="300" height="300">
### Analysis
Firstly, Among the tested output, since the denoize time will be affected by the various variables, I analyzed the denoize in the default condition given to us:
800x800 resolution, 80 filter size. The output is: 4.645Ms


Based on my observation, with the denoize feature implemented, the iteration amount needed for reaching the acceptable smooth condition was significantly reduced, leading to more efficient and time-saving results in the process.

Also, based on my test I got the following graph among the resolution and runtime of denoize:

|Graph1|
|:-----:|
|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/denoize_resolution.png" width="700" height="500">



It is easy to see that while the resolution increased, the amount of that denoize needed was increased. It is also easy to understand: there are more pixels needed to estimate, and thus takes more time to denoize.

Additionally, I get the following graph among the filter size and runtime of denoize:

|Graph2|
|:-----:|
|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/resolution.png" width="700" height="500">

It is easy to see from the graph, that the runtime needed increased a little bit while the filter size increased. But the increase is rather small. For the small wave in the graph, while the filter size increases, I guess it is because the increased amount is small and even hidden.


I recorded the visual output for different filter sizes and get the following graph:
(For comparing, the iterations was set to 100)

|0|10|20|30|40|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-21_00-28-26z.100samp.png" width="300" height="200">|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-21_00-33-12z.100samp.png" width="200" height="200">|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-21_00-33-40z.100samp.png" width="200" height="200">|<img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-21_00-34-03z.100samp.png" width="200" height="300" >| <img src="https://github.com/Ibm510000/Project4-CUDA-Denoiser/blob/base-code/img/cornell.2023-10-21_00-34-22z.100samp.png" width="200" height="300" >






