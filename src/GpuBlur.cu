#include <cuda_runtime.h>
#include <stdio.h>
#include "GpuBlur.hpp"

// 15×15 Gaussian kernel storage in constant memory
#define GAUSSIAN_KERNEL_RADIUS 7
#define GAUSSIAN_KERNEL_SIZE  (2 * GAUSSIAN_KERNEL_RADIUS + 1)
static __constant__ float d_gaussKernel[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];

// A simple clamp function.
__device__ unsigned char clamp255(float value) {
    if (value > 255.0f) return 255;
    if (value < 0.0f)   return 0;
    return static_cast<unsigned char>(value);
}

// The CUDA kernel for a 15×15 Gaussian blur (BGR).
__global__ void gaussianBlurKernel(const unsigned char* inData, unsigned char* outData,
                                   int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sumB = 0.f, sumG = 0.f, sumR = 0.f;

    // Loop over the 15×15 kernel
    for (int ky = -GAUSSIAN_KERNEL_RADIUS; ky <= GAUSSIAN_KERNEL_RADIUS; ky++) {
        for (int kx = -GAUSSIAN_KERNEL_RADIUS; kx <= GAUSSIAN_KERNEL_RADIUS; kx++) {
            int px = x + kx;
            int py = y + ky;

            // Boundary check
            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            if (py < 0) py = 0;
            if (py >= height) py = height - 1;

            int inIdx = (py * width + px) * channels;

            // Compute index into the Gaussian kernel
            int kernelX = kx + GAUSSIAN_KERNEL_RADIUS;
            int kernelY = ky + GAUSSIAN_KERNEL_RADIUS;
            float w = d_gaussKernel[kernelY * GAUSSIAN_KERNEL_SIZE + kernelX];

            sumB += w * inData[inIdx + 0];
            sumG += w * inData[inIdx + 1];
            sumR += w * inData[inIdx + 2];
        }
    }

    int outIdx = (y * width + x) * channels;
    outData[outIdx + 0] = clamp255(sumB);
    outData[outIdx + 1] = clamp255(sumG);
    outData[outIdx + 2] = clamp255(sumR);
}

// -------------------------
// Public API from .hpp
// -------------------------

// Run the Gaussian blur kernel
void runBlurKernel(const unsigned char* inData, unsigned char* outData,
                   int width, int height, int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    gaussianBlurKernel<<<grid, block>>>(inData, outData, width, height, channels);
    cudaDeviceSynchronize();
}

// Build a 15×15 Gaussian kernel on CPU, then copy to GPU constant memory
void uploadGaussianKernel(float sigma)
{
    const int kSize = GAUSSIAN_KERNEL_SIZE;
    float hKernel[kSize * kSize];
    float sum = 0.0f;

    // If sigma=0, pick a default
    if (sigma <= 0.f) {
        // A typical heuristic from OpenCV for kernel = 15
        // This is approximate (OpenCV's exact formula is a bit more involved)
        sigma = 0.3f * ((kSize - 1.f) * 0.5f - 1.f) + 0.8f;
    }

    int center = kSize / 2;
    for (int y = 0; y < kSize; y++) {
        for (int x = 0; x < kSize; x++) {
            int rx = x - center;
            int ry = y - center;

            float val = expf(-(rx * rx + ry * ry) / (2.0f * sigma * sigma));
            hKernel[y * kSize + x] = val;
            sum += val;
        }
    }

    // Normalize so that sum of all weights = 1
    for (int i = 0; i < kSize * kSize; i++) {
        hKernel[i] /= sum;
    }

    // Copy to GPU constant memory
    cudaMemcpyToSymbol(d_gaussKernel, hKernel, sizeof(hKernel));
}
