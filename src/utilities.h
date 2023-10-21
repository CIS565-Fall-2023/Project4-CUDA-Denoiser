#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define PI_OVER_2         1.57079632679489661923f
#define PI_OVER_4         0.78539816339744830961f 
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

// Easing how much I have to write when using unique pointers
#define uPtr std::unique_ptr
#define mkU std::make_unique
#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2

#define SORT_BY_MATERIAL 0
#define STREAM_COMPACT 0
#define CACHE_FIRST_INTERSECTION 1
#define ENABLE_NAIVE_AABB_OPTIMISATION 1
#define ENABLE_BVH 1
#define ENABLE_RUSSIAN_ROULETTE 1
#define ENABLE_HDR_GAMMA_CORRECTION 0

#define GAMMA 2.2

#define DEBUG 1

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

enum class RenderMode
{
    FULL,
    POSITIONS,
    NORMALS,
    DEPTH,
    COUNT
};

enum class DenoiseMode
{
    NONE,
    DENOISE_AFTER_PATHTRACING,
    DENOISE_AT_EVERY_ITERATION,
    COUNT
};

struct DenoiserParameters
{
    DenoiseMode mode;
    int maxIters;
    int denoiseIters;
    float phiCol;
    float phiPos;
    float phiNor;
};

/**
* This class is used for timing the performance
* Uncopyable and unmovable
*
* Adapted from WindyDarian(https://github.com/WindyDarian)
*/
class PerformanceTimer
{
public:
    PerformanceTimer()
    {
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_end);
    }

    ~PerformanceTimer()
    {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_end);
    }

    void startCpuTimer()
    {
        if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
        cpu_timer_started = true;

        time_start_cpu = std::chrono::high_resolution_clock::now();
    }

    void endCpuTimer()
    {
        time_end_cpu = std::chrono::high_resolution_clock::now();

        if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

        std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
        prev_elapsed_time_cpu_milliseconds =
            static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

        cpu_timer_started = false;
    }

    void startGpuTimer()
    {
        if (gpu_timer_started) { cudaEventRecord(event_end); /*throw std::runtime_error("GPU timer already started");*/ }
        gpu_timer_started = true;

        cudaEventRecord(event_start);
    }

    void endGpuTimer()
    {
        cudaEventRecord(event_end);
        cudaEventSynchronize(event_end);

        //if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

        cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
        gpu_timer_started = false;
    }

    float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
    {
        return prev_elapsed_time_cpu_milliseconds;
    }

    float getGpuElapsedTimeForPreviousOperation() //noexcept
    {
        return prev_elapsed_time_gpu_milliseconds;
    }

    // remove copy and move functions
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer(PerformanceTimer&&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

private:
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;

    using time_point_t = std::chrono::high_resolution_clock::time_point;
    time_point_t time_start_cpu;
    time_point_t time_end_cpu;

    bool cpu_timer_started = false;
    bool gpu_timer_started = false;

    float prev_elapsed_time_cpu_milliseconds = 0.f;
    float prev_elapsed_time_gpu_milliseconds = 0.f;
};