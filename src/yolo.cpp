#include "yolo.h"
#include <iostream>
#include <math.h>

void Yolo::GenerateGridsAndStride(std::vector<int>& strides, int* gridStrides)
{
    for (auto stride : strides)
    {
        int num_grid_y = inputH / stride;
        int num_grid_x = inputW / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                *gridStrides++ = g0;
                *gridStrides++ = g1;
                *gridStrides++ = stride;
            }
        }
    }
}

float Yolo::BoxIou(float ax1, float ay1, float aw, float ah,
              float bx1, float by1, float bw, float bh)
{
    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax1 + aw, bx1 + bw);
    float inter_y2 = std::min(ay1 + ah, by1 + bh);

    float inter_area = std::max(inter_x2 - inter_x1, 0.0f) * std::max(inter_y2 - inter_y1, 0.0f);
    float box1_area = aw * ah;
    float box2_area = bw * bh;
    float union_area = box1_area + box2_area - inter_area;
    if (union_area <= 0.0f)
        return 0.0f;
    return inter_area / union_area;
}

void Yolo::NMS(float* objects, float iouThresh, int width)
{
    // objects 1 + [8400, 7] 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    int count = objects[0];
    for (int i = 0; i < count; ++i)
    {
        float* pcurrent = objects + 1 + i * width;
        if (pcurrent[6] == 0)
            continue;
        for (int j = 0; j < count; ++j)
        {
            float* pitem = objects + 1 + j * width;
            if (pitem[6] == 0 || i == j || pitem[5] != pcurrent[5])
                continue;
            
            if (pitem[4] > pcurrent[4])
            {
                if (pitem[4] == pcurrent[4] && j < i)
                    continue;
                
                float iou = BoxIou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                              pitem[0], pitem[1], pitem[2], pitem[3]);
                if (iou > iouThresh)
                {
                    pcurrent[6] = 0;    // 1=keep, 0=ignore
                    objects[0] -= 1;    // keepFlag count -1
                    break;
                }
            }
        }
    }
}

void Yolo::GenerateYoloProposals(int* gridStrides, int gridStrideSize, float* outputSrc, 
float bboxConfThresh, float* objects, int numClass)
{
    // outputSrc: [bs, 8400, 4 + 1 + NumClasses]
    // objects: 1 + [8400, 7] 1: save keepFlag count 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    int count = 0;
    for (int anchorIdx = 0; anchorIdx < gridStrideSize; anchorIdx++)
    {
        int x = gridStrides[3*anchorIdx];
        int y = gridStrides[3*anchorIdx+1];
        int stride = gridStrides[3*anchorIdx+2];
        // psrc = outputSrc[anchorIdx]
        float* psrc = outputSrc + anchorIdx * (4 + 1 + numClass);

        float box_objectness = psrc[4];

        for (int classIdx = 0; classIdx < numClass; classIdx++)
        {
            float box_cls_score = psrc[5 + classIdx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > bboxConfThresh)
            {
                float* pobject = objects + 1 + count * 7;

                // center x, y, w, h
                float x0 = (x + psrc[0]) * stride;
                float y0 = (y + psrc[1]) * stride;
                float w = exp(psrc[2]) * stride;
                float h = exp(psrc[3]) * stride;
                // caculate left top 
                x0 = x0 - w * 0.5f;
                y0 = y0 - h * 0.5f;
                pobject[0] = x0;
                pobject[1] = y0;
                pobject[2] = w;
                pobject[3] = h;
                pobject[4] = box_prob;
                pobject[5] = classIdx;
                pobject[6] = 1;
                count++;
            }
        }
    }
    objects[0] = count;
}

std::vector<Object> Yolo::DecodeOutput(float* outputSrc, float scale, const int img_w, const int img_h)
{   
    mOutputSrc = outputSrc;
    GenerateYoloProposals(gridStrides, gridStrideSize, outputSrc, 
    bboxConfThresh, mOutputObject, NumClasses);

    int count = mOutputObject[0];
    std::cout << "num of boxes before nms: " << mOutputObject[0] << std::endl;

    NMS(mOutputObject, iouThresh, mOutputWidth);
    std::cout << "num of boxes after nms: " << mOutputObject[0] << std::endl;

    std::vector<Object> objects;
    for (int i = 0; i < count; ++i)
    {
        float* pobject = mOutputObject + 1 + i * mOutputWidth;
        if (pobject[6] == 0)
            continue;
        float x0 = pobject[0] / scale;
        float y0 = pobject[1] / scale;
        float x1 = (pobject[0] + pobject[2]) / scale;
        float y1 = (pobject[1] + pobject[3]) / scale;
        // clip 
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        Object object;
        object.x = x0;
        object.y = y0;
        object.w = x1 - x0;
        object.h = y1 - y0;
        object.prob = pobject[4];
        object.label = pobject[5];
        objects.push_back(object);
    }
    return objects;
}

Yolo::Yolo(int inputH, int inputW, int NumClasses, float bboxConfThresh, float iouThresh)
{
    this->NumClasses = NumClasses;
    this->inputH = inputH;
    this->inputW = inputW;
    this->mOutputWidth = 7;
    this->mOutputSrc = new float[8400 * (4 + 1 + NumClasses)];
    this->mOutputObject = new float[1 + 8400 * (4 + 1 + NumClasses)];
    this->bboxConfThresh = bboxConfThresh;
    this->iouThresh = iouThresh;
    // generate grid strides
    for (auto stride : this->strides)
    {
        int num_grid_y = inputH / stride;
        int num_grid_x = inputW / stride;
        this->gridStrideSize += num_grid_y * num_grid_x;
    }
    this->gridStrides = new int[this->gridStrideSize * this->strides.size()];
    this->GenerateGridsAndStride(this->strides, this->gridStrides);
}

Yolo::~Yolo()
{
    std::cout << "===> Destroy Yolo instance" <<std::endl;
    delete[] this->gridStrides;
    // delete[] this->mOutputSrc;
    delete[] this->mOutputObject;
}