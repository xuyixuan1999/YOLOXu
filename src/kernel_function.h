#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


void blobFromImageCuda(float* blobDev, const cv::Mat& img, const cudaStream_t& stream = 0);

void GenerateYoloProposalDevice(int* gridStrides, int gridStrideSize, float* outputSrc, float* objects,
                            float bboxConfThresh, int numClass, const cudaStream_t& stream = 0);

void FastNMSDevice(float* objects, float iou_threshold, int width, int topK, const cudaStream_t& stream = 0);