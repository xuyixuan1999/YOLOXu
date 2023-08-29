#pragma once
#include <vector>
#include <NvInfer.h>

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
    void GenerateYoloProposals(int* gridStrides, int gridStrideSize, float* outputSrc, 
                            float bboxConfThresh, float* objects, int numClass);
    std::vector<Object> DecodeOutput(float* outputSrc, float scale, const int img_w, const int img_h);

private:
    bool isDevice;
    // params for yolo
    int NumClasses;
    int inputW;
    int inputH;
    int* gridStridesHost = nullptr; // [8400, 3]
    float* mOutputSrcHost = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    int mOutputWidth = 7; // 7:left, top, right, bottom, confidence, class, keepflag; 
    float* mOutputObjectHost = nullptr; // 1 + [bs, 8400, 7]
    float bboxConfThresh;
    float iouThresh;
    int gridStrideSize;;

    int* gridStridesDevice = nullptr; // [8400, 3]
    float* mOutputSrcDevice = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    float* mOutputObjectDevice = nullptr; // 1 + [bs, 8400, 7]
    
    std::vector<int> strides = {8, 16, 32};

public:
    Yolo(int inputH, int inputW, int NumClasses, float bboxConfThresh, float iouThresh, bool isDevice);
    ~Yolo();
};