#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "utils.h"


void blobFromImageCuda(float* blobDev, const cv::Mat& img, const cudaStream_t& stream = 0);

void GenerateYoloProposalDevice(int* gridStrides, int gridStrideSize, float* outputSrc, float* objects, float bboxConfThresh, int numClass, const cudaStream_t& stream = 0);

void FastNMSDevice(float* objects, float iou_threshold, int width, int topK, const cudaStream_t& stream = 0);

void ResizePaddingDevice(uchar* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, float scale, AffineMatrix d2s, const cudaStream_t& stream = 0); 

void HWC2CHWDevice(float* src, float* dst, int width, int height, int channels, const cudaStream_t& stream = 0);