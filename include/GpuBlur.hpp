#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    // Optionally, provide a way to set or update the Gaussian kernel from CPU:
    void uploadGaussianKernel(float sigma);
    void runBlurKernel(const unsigned char* inData, unsigned char* outData,
                       int width, int height, int channels);

#ifdef __cplusplus
}
#endif
