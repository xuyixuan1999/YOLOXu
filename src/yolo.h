#pragma once
#include <vector>
#include <NvInfer.h>
#include "kernel_function.h"
#include "utils.h"

class Yolo
{
public:
    float BoxIou(float ax1, float ay1, float aw, float ah, float bx1, float by1, float bw, float bh);
    void GenerateGridsAndStride(std::vector<int>& strides, int* grid_strides);
    void NMS(float* objects, float iou_threshold, int width);
    void GenerateYoloProposals(int* gridStrides, int gridStrideSize, float* outputSrc, float bboxConfThresh, float* objects, int numClass);
    
    void PreProcess(cv::Mat img);
    void PostProcess(std::vector<Object>& objects);
    
    void PreProcessDevice(cv::Mat, cudaStream_t stream = 0);
    void PostProcessDevice(std::vector<Object>& objects, const cudaStream_t& stream = 0);
    

    float* mInputCHWDevice = nullptr; // [c, h, w]
    float* mInputCHWHost = nullptr; // [c, h, w]
    float* mOutputSrcDevice = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    float* mOutputSrcHost = nullptr; // [bs, 8400, 4 + 1 + num_classes]

private:
    // params for yolo  
    bool isDevice;
    int NumClasses;
    int topK = 200;
    float bboxConfThresh;
    float iouThresh;
    int gridStrideSize;
    int mOutputWidth = 7; // 7:left, top, right, bottom, confidence, class, keepflag; 
    std::vector<int> strides = {8, 16, 32};
    AffineMatrix md2s;

    // input
    int srcW;
    int srcH;
    cv::Size mInputSize;
    unsigned char* mInputDevice = nullptr; // [srcH, srcW, c]
    float* mInputResizeDevice = nullptr; // [dstH, dstW, c]
    float mScale;
    
    // output 
    int dstW;
    int dstH;
    // host memory
    int* gridStridesHost = nullptr; // [8400, 3]
    float* mOutputObjectHost = nullptr; // 1 + [bs, 8400, 7] 1: save keepFlag count, 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    // device memory
    int* gridStridesDevice = nullptr; // [8400, 3]
    float* mOutputObjectDevice = nullptr; // 1 + [bs, 8400, 7]

public:
    Yolo(int inputH, int inputW, int NumClasses, float bboxConfThresh, float iouThresh, bool isDevice);
    ~Yolo();
};