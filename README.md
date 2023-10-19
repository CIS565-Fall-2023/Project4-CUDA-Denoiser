CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Helena Zhang
* Tested on: Windows 11, i7-10750 @ 2.6GHz 16GB, Geforce RTX 2060 6GB

### Analysis

(Todo):
 * How much time denoising adds to rendering: Nsight compute overall process & atrous kernel
 * how denoising influences the number of iterations needed to get an "acceptably smooth" result
 * how denoising at different resolutions impacts runtime: nsight compute # of pixels vs runtime
 * how varying filter sizes affect performance: nsight compute filter size vs runtime
 * how visual results vary with filter size -- does the visual quality scale uniformly with filter size?
 * how effective/ineffective is this method with different material types
 * how do results compare across different scenes - for example, between cornell.txt and cornell_ceiling_light.txt. Does one scene produce better denoised results? Why or why not?
