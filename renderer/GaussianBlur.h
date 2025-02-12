#pragma once

#include "for_each.h"
#include <vector>
#include <cmath>

#include "visionaray/detail/thread_pool.h"

namespace visionaray {

void gaussianBlurHorizontal(thread_pool& threadPool,
                            uint8_t* input,
                            uint8_t* output,
                            uint2 size,
                            const float* kernel,
                            size_t kernelSize)
{
#ifdef WITH_CUDA
  cuda::for_each(0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  hip::for_each(0, size.x, 0, size.y,
#else
  parallel::for_each(threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {
        float sum[4] = {0, 0, 0, 0};
        float weightSum = 0.0f;
          
        int halfSize = kernelSize / 2;
          
        for (int i = -halfSize; i <= halfSize; i++)
        {
          int xi = std::min((int)size.x - 1, std::max(x + i, 0));
          auto pixelIdx = 4 * (y * size.x + xi);
          float weight = kernel[i + halfSize];
      
          sum[0] += input[pixelIdx + 0] * weight;
          sum[1] += input[pixelIdx + 1] * weight;
          sum[2] += input[pixelIdx + 2] * weight;
          sum[3] += input[pixelIdx + 3] * weight;
      
          weightSum += weight;
        }
      
        output[4 * (y * size.x + x) + 0] = std::min(255, std::max(0, int(sum[0] / weightSum)));
        output[4 * (y * size.x + x) + 1] = std::min(255, std::max(0, int(sum[1] / weightSum)));
        output[4 * (y * size.x + x) + 2] = std::min(255, std::max(0, int(sum[2] / weightSum)));
        output[4 * (y * size.x + x) + 3] = std::min(255, std::max(0, int(sum[3] / weightSum)));
      });
}

void transposeImage(thread_pool& threadPool, const uint8_t* input, uint8_t* output, uint2 size)
{
#ifdef WITH_CUDA
  cuda::for_each(0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  hip::for_each(0, size.x, 0, size.y,
#else
  parallel::for_each(threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {
          for (int i=0; i<4; ++i)
            output[4 * (x * size.y + y) + i] = input[4 * (y * size.x + x) + i];
      });
}

std::vector<float> computeGaussianKernel(float sigma)
{
  int kernelSize = 2 * ceil(3 * sigma) + 1;
  int halfSize = kernelSize / 2;
  std::vector<float> kernel(kernelSize);

  float sum = 0.0f;
  for (int i = -halfSize; i <= halfSize; i++)
  {
    kernel[i + halfSize] = expf(-0.5f * (i * i) / (sigma * sigma));
    sum += kernel[i + halfSize];
  }

  for (float &value : kernel)
  {
    value /= sum;
  }

  return kernel;
}

static std::vector<float> lastGaussianKernel;
static float lastSigma{-1.f};

// expects RGBA (8888) image
void applyGaussianBlur(thread_pool& threadPool, uint8_t* onDeviceInput, uint8_t* onDeviceOutput, uint2 size, float sigma)
{
  if (sigma != lastSigma)
  {
    lastSigma = sigma;
    lastGaussianKernel = computeGaussianKernel(sigma);
  }
  std::vector<float>& kernel = lastGaussianKernel;
  auto kernelSize = kernel.size();
  float* onDeviceKernel;
  uint8_t* onDeviceTemp1;
  uint8_t* onDeviceTemp2;

#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaMalloc(&onDeviceKernel, kernelSize * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(onDeviceKernel, kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc(&onDeviceTemp1, size.x * size.y * 4));
  CUDA_SAFE_CALL(cudaMalloc(&onDeviceTemp2, size.x * size.y * 4));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipMalloc(&onDeviceKernel, kernelSize * sizeof(float)));
  HIP_SAFE_CALL(hipMemcpy(onDeviceKernel, kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice));
  HIP_SAFE_CALL(hipMalloc(&onDeviceTemp1, size.x * size.y * 4));
  HIP_SAFE_CALL(hipMalloc(&onDeviceTemp2, size.x * size.y * 4));
#else
  onDeviceKernel = kernel.data();
  auto temp1 = std::vector<uint8_t>(size.x * size.y * 4);
  auto temp2 = std::vector<uint8_t>(size.x * size.y * 4);
  onDeviceTemp1 = temp1.data();
  onDeviceTemp2 = temp2.data();
#endif

  gaussianBlurHorizontal(threadPool, onDeviceInput, onDeviceTemp1, size, onDeviceKernel, kernelSize);
  transposeImage(threadPool, onDeviceTemp1, onDeviceTemp2, size);
  gaussianBlurHorizontal(threadPool, onDeviceTemp2, onDeviceTemp1, uint2{size.y, size.x}, onDeviceKernel, kernelSize);
  transposeImage(threadPool, onDeviceTemp1, onDeviceOutput, uint2{size.y, size.x});

#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(onDeviceTemp1));
  CUDA_SAFE_CALL(cudaFree(onDeviceTemp2));
  CUDA_SAFE_CALL(cudaFree(onDeviceKernel));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipFree(onDeviceTemp1));
  HIP_SAFE_CALL(hipFree(onDeviceTemp2));
  HIP_SAFE_CALL(hipFree(onDeviceKernel));
#endif
}

} // namespace visionaray
