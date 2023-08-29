#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include "kernel_function.h"

__global__ 
void blobFromImageKernel(const uchar* imgData, float* blob, int channels, int img_h, int img_w) 
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (h < img_h && w < img_w) {
        for (int c = 0; c < channels; ++c)
        {
            blob[c * img_w * img_h + h * img_w + w] = static_cast<float>(imgData[h * img_w * channels + w * channels + c]);
        }
    }
}

__global__
void decodeOutputKernel(float* prob, float scale, int img_h, int img_w)
{
    // coming soon
}

void blobFromImageCuda(float* blobDev, const cv::Mat& img, const cudaStream_t& stream) 
{
    int channels = img.channels();
    int img_h = img.rows;
    int img_w = img.cols;
    int64_t blobSize = channels * img_h * img_w * sizeof(float);

    // Copy image data from CPU to GPU using the provided stream
    uchar* imgDataDev = nullptr;
    cudaMalloc((void**)&imgDataDev, img.total() * img.channels() * sizeof(uchar));
    cudaMemcpyAsync(imgDataDev, img.data, img.total() * img.channels() * sizeof(uchar), cudaMemcpyHostToDevice, stream);

    dim3 blockSize(16, 16);
    dim3 gridSize((img_w + blockSize.x - 1) / blockSize.x, (img_h + blockSize.y - 1) / blockSize.y);

    blobFromImageKernel<<<gridSize, blockSize, 0, stream>>>(imgDataDev, blobDev, channels, img_h, img_w);

    cudaFree(imgDataDev);  // Free GPU memory used for image data
}
