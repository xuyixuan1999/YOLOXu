#pragma once
#include <vector>
#include <NvInfer.h>
#include "kernel_function.h"

struct Object
{
    // cv::Rect_<float> rect;
    float x, y, w, h;  // center and width, height
    int label;
    float prob;
};

class Yolo
{
public:
    void GenerateGridsAndStride(std::vector<int>& strides, int* grid_strides);
    float BoxIou(float ax1, float ay1, float aw, float ah,
                 float bx1, float by1, float bw, float bh);
    void NMS(float* objects, float iou_threshold, int width);
    void GenerateYoloProposals(int* gridStrides, int gridStrideSize, float* outputSrc, float bboxConfThresh, float* objects, int numClass);
    void DecodeOutput(std::vector<Object>& objects, float* outputSrc, float scale, const int img_w, const int img_h);

    void DecodeOutputDevice(std::vector<Object>& objects, float scale, const int img_w, const int img_h, const cudaStream_t& stream = 0);

    float* tmpOutputSrc = nullptr;

private:
    bool isDevice;
    // params for yolo
    int NumClasses;
    int inputW;
    int inputH;
    int topK = 200;

    float bboxConfThresh;
    float iouThresh;
    int gridStrideSize;
    int mOutputWidth = 7; // 7:left, top, right, bottom, confidence, class, keepflag; 
    // host memory
    int* gridStridesHost = nullptr; // [8400, 3]
    float* mOutputSrcHost = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    float* mOutputObjectHost = nullptr; // 1 + [bs, 8400, 7] 1: save keepFlag count, 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    // device memory
    int* gridStridesDevice = nullptr; // [8400, 3]
    float* mOutputSrcDevice = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    float* mOutputObjectDevice = nullptr; // 1 + [bs, 8400, 7]
    
    std::vector<int> strides = {8, 16, 32};

public:
    Yolo(int inputH, int inputW, int NumClasses, float bboxConfThresh, float iouThresh, bool isDevice);
    ~Yolo();
};