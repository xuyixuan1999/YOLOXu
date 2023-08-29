#pragma once
#include <vector>

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
    void GenerateYoloProposals(int* grid_strides, int grid_strides_size, float* outputSrc, float bbox_conf_thresh, float* objects, int num_class);
    std::vector<Object> DecodeOutput(float* outputSrc, float scale, const int img_w, const int img_h);

private:
    // params for yolo
    int NumClasses;
    int inputW;
    int inputH;
    int* gridStrides = nullptr; // [8400, 3]
    float* mOutputSrc = nullptr; // [bs, 8400, 4 + 1 + num_classes]
    int mOutputWidth; // 7:left, top, right, bottom, confidence, class, keepflag; 
    float* mOutputObject = nullptr; // 1 + [bs, 8400, 7]
    float bboxConfThresh;
    float iouThresh;
    int gridStrideSize;;
    
    std::vector<int> strides = {8, 16, 32};

public:
    Yolo(int input_h, int input_w, int num_classes, float bbox_conf_thresh, float iou_threshold);
    ~Yolo();
};