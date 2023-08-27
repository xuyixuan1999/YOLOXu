#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

float* blobFromImageCuda(const cv::Mat& img, const cudaStream_t& stream = 0);