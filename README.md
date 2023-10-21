CUDA Denoiser For CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xiaoyue Ma
  * [LinkedIn](https://www.linkedin.com/in/xiaoyue-ma-6b268b193/)
* Tested on: Windows 10, i7-12700H @ 2.30 GHz 16GB, GTX3060 8GB

# Overview
This project implements a CUDA-powered pathtracing denoiser that utilizes geometry buffers (G-buffers) to steer a smoothing filter. Drawing inspiration from the study "[Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)", this denoiser enhances the appearance of pathtraced images by producing smoother results with fewer samples per pixel.

<p align="center">
  <img width="800" height="280" src="img/alg.png" alt="Intro">
</p> 

By iteratively applying sparse blurs of increasing sizes, one can approximate the effects of a Gaussian filter. Instead of using a large filter, this method utilizes a smaller one but spaces out the samples that pass through it. Interestingly, to achieve a greater blur size, it doesn't require enlarging the filter; instead, it demands more iterations. This approach optimizes performance while still achieving the desired blurring effect.

<p align="center">
  <img width="800" height="200" src="img/alg2.png" alt="Intro">
</p> 

# Features
## Core features
##### **A-Trous**

A-Trous Filtering enhances our denoiser's efficiency by applying iterative sparse blurs, achieving large filter results with a smaller filter size.

|No Filter | Filter Size = 16 | Filter Size = 64 |
|:-----: | :-----: |:-----: |
|![](img/10ori.png) | ![](img/10GnoEdge16.png) | ![](img/10AnoEdge64.png) |

##### **Edge-Avoiding A-Trous**

A-Trous Filtering can blur vital details, but using G-buffer information, we adjust blurring on sharp edges to preserve key image elements, optimizing denoising.

|No Filter | A-Trous (64) | A-Trous with Edge-Avoiding (64) |
|:-----: | :-----: |:-----: |
|![](img/10ori.png) | ![](img/10AnoEdge64.png) | ![](img/10AEdge64.png) |

##### **G-Buffer**

Using normal, position, and time intersection data as weights, we minimize edge blurring during application. This data is viewable via the "Show GBuffer" option in the GUI.

| Time to Intersect |Normal | Position |
|:-----: | :-----: |:-----: |
|![](img/time.png) | ![](img/pos.png) | ![](img/nor.png) |

## Extra Feature

#### Gaussian Filtering


The Gaussian Filter calculates a pixel's new color by averaging its neighbors, giving more weight to those nearby. In my tests, it generated a more blurred image when edge-avoiding was off and a marginally noisy one when on.

|No Filter | A-Trous (16)  | Gaussian (16)|
|:-----: | :-----: |:-----: |
|![](img/10ori.png) | ![](img/10AnoEdge16.png) | ![](img/10GnoEdge16.png) |


| A-Trous Edge-Avoiding(64) | Gaussian Edge-Avoiding(64)|
|:-----: | :-----:|
![](img/10AEdge64.png) | ![](img/10GEdge64.png) |



# Performance Analysis

### **How much time denoising adds to the renders**

The denoiser activates after the path tracer's image rendering, with its runtime being influenced by image resolution and filter size, not scene complexity. My tests show that for an 800x800 image with an 80x80 filter, the denoising time remains consistent regardless of iteration count.   

<p align="center">
  <img width="800" height="400" src="img/chart1.png" alt="Chart">
</p> 


### **How denoising influences the number of iterations needed to get an "acceptably smooth" result**

Perceptions of "smoothness" differ among individuals and can be influenced by various image factors. In the **'cornell_ceiling_light'** test, a smooth appearance was achieved at 300 iterations without denoising. With denoising, only 150 iterations were needed, marking a 100% reduction. While the benefit of denoising can depend on the scene, it notably reduces required iterations overall.

No Denoising, 300 iterations | Denoised, 150 iterations
:----------:|:-----------:
![](img/300ori.png) | ![](img/150de.png)

### **How denoising at different resolutions impacts runtime**

The runtime for denoising rises with resolution, but the increase isn't linear. For instance, even though a significant resolution jump (from 200x200 to 800x800) is made, the runtime only multiplies by seven. As the resolution increases, the denoising process requires more time, attributed to the higher pixel count and added A-trous filter iterations. However, this growth in runtime doesn't scale proportionally with the resolution.

<p align="center">
  <img width="800" height="500" src="img/chart2.png" alt="Chart">
</p> 


### **How varying filter sizes affect performance**

Predictably, an increase in filter size results in an extended denoising runtime because more neighboring pixels get sampled to determine each pixel's new color. In a chart derived from the test at an 800x800 resolution, while there's a relationship between additional time and filter size, it's not strictly linear.

<p align="center">
  <img width="800" height="500" src="img/chart3.png" alt="Chart">
</p> 

### **How visual results vary with filter size -- does the visual quality scale uniformly with filter size?**

For images from the test scene at 10 iteration, denoising sees a marked improvement from 32 to 64 and a discernible one from 16 to 32. However, further increments offer limited visual benefits. 

Filter Size 4 | Filter Size 16 | Filter Size 32 | Filter Size 64 |
:-----:|:-----:|:-----:|:-----:|
![](img/10AEdge4.png)  | ![](img/10AEdge16.png) | ![](img/10AEdge32.png) | ![](img/10AEdge64.png) | 

### **How effective/ineffective is this method with different material types**   

This technique excels with diffuse materials, but often results in a softer appearance as the "roughness" diminishes from color smoothing. Its efficacy is diminished with reflections, causing them to blur noticeably and reducing the material's shine.

Diffuse | Specular | Imperfect Specular
:----------:|:-----------:|:-----------:
![](img/diffuse.png) | ![](img/10AEdge64.png)  | ![](img/imperfact.png) 


### **How do results compare across different scenes**
**For example, between `cornell.txt` and `cornell_ceiling_light.txt`. Does one scene produce better denoised results? Why or why not?**

The results across different scenes vary greatly. For example, the denoiser works exceptionally on the `cornell_ceiling_light` scene, but not so much on the regular `cornell` scene.    
From my testing, denoiser seems to work better on bright scenes. As I dig deeper, I realize that it's not actually the brightness, but the color variations. In a bright scene, most pixels are lit up uniformly and the LTE computation converges more quickly. On the other hand, when the scene is dark, the bright pixels are more sparse and there are inherently more noises in the scene, which makes denoising a harder task.     
Since we are using normals/positions/time to intersect to avoid the edges, when the different edges actually have different normal/position/time, our algorithm will expectedly work better.    
It is worth mentioning that different scenes also require different norm/pos/t configurations to look the best.   

Cornell Scene | Light Cornell Scene 
:----------:|:-----------:
![](img/10cornell.png) | ![](img/10AEdge64.png) 


### **A-Trous Filtering vs. Gaussian Filtering**

For performance comparison, as expected, A-Trous Filtering outperforms Gaussian Filtering significantly. Specifically, the performance of A-Trous and Gaussian are comparable with a filter size of 10 (resolution 800x800), but the runtime of A-Trous increases almost linearly whereas the runtime of Gaussian increases exponentially. They are also comparable at very small resolution, but again the runtime of Gaussian increases exponentially with the resolution whereas A-Trous only increases linearly.    
This makes perfect sense since A-Trous algorithm always takes 5x5 samples for each pixel, and only increase the number of iterations when the filter size increases. However, Gaussian blur takes nxn samples, which is a exponential increase.     

<p align="center">
  <img width="800" height="500" src="img/chart4.png" alt="Chart">
</p> 

