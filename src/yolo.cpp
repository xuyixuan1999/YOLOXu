#include "yolo.h"
#include <iostream>
#include <math.h>
#include "error.cuh"

void Yolo::GenerateGridsAndStride(std::vector<int>& strides, int* gridStridesHost)
 {

    for (auto stride : strides)
    {
        int num_grid_y = dstH / stride;
        int num_grid_x = dstW / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                *gridStridesHost++ = g0;
                *gridStridesHost++ = g1;
                *gridStridesHost++ = stride;
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

        for (int j = 0; j < count; ++j)
        {
            float* pitem = objects + 1 + j * width;
            if ( i == j || pitem[5] != pcurrent[5])
                continue;
            
            if (pitem[4] >= pcurrent[4])
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

void Yolo::GenerateYoloProposals(int* gridStridesHost, int gridStrideSize, float* outputSrc, 
float bboxConfThresh, float* objects, int numClass)
{
    // outputSrc: [bs, 8400, 4 + 1 + NumClasses]
    // objects: 1 + [8400, 7] 1: save keepFlag count 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    int count = 0;
    for (int anchorIdx = 0; anchorIdx < gridStrideSize; anchorIdx++)
    {
        int x = gridStridesHost[3*anchorIdx];
        int y = gridStridesHost[3*anchorIdx+1];
        int stride = gridStridesHost[3*anchorIdx+2];
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

void Yolo::PreProcess(cv::Mat img)
{
    if (img.size() != mInputSize)
    {
        srcW = img.cols;
        srcH = img.rows;
        mInputSize = cv::Size(srcW, srcH);
        mScale = std::min(dstW / (img.cols*1.0), dstH / (img.rows*1.0));
    }
    cv::Mat prImage = static_resize(img, dstW, dstH);
    blobFromImage(mInputCHWHost, prImage);
}

void Yolo::PostProcess(std::vector<Object>& objects)
{   
    // clear keepFlag and count
    memset(mOutputObjectHost, 0, (1 + gridStrideSize * mOutputWidth) * sizeof(float));
    objects.clear();
    GenerateYoloProposals(gridStridesHost, gridStrideSize, mOutputSrcHost, 
    bboxConfThresh, mOutputObjectHost, NumClasses);

    int count = mOutputObjectHost[0];
    // std::cout << "num of boxes before nms: " << mOutputObjectHost[0] << std::endl;

    NMS(mOutputObjectHost, iouThresh, mOutputWidth);
    // std::cout << "num of boxes after nms: " << mOutputObjectHost[0] << std::endl;

    for (int i = 0; i < count; ++i)
    {
        float* pobject = mOutputObjectHost + 1 + i * mOutputWidth;
        if (pobject[6] == 0)
            continue;
        float x0 = pobject[0] / mScale;
        float y0 = pobject[1] / mScale;
        float x1 = (pobject[0] + pobject[2]) / mScale;
        float y1 = (pobject[1] + pobject[3]) / mScale;
        // clip 
        x0 = std::max(std::min(x0, (float)(srcW - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(srcH - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(srcW - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(srcH - 1)), 0.f);

        Object object;
        object.x = x0;
        object.y = y0;
        object.w = x1 - x0;
        object.h = y1 - y0;
        object.prob = pobject[4];
        object.label = pobject[5];
        objects.push_back(object);
    }
}

void Yolo::PreProcessDevice(cv::Mat img, cudaStream_t stream)
{
    if (img.size() != mInputSize)
    {
        srcW = img.cols;
        srcH = img.rows;
        mInputSize = cv::Size(srcW, srcH);
        // malloc mInputDevice
        CHECK(cudaMalloc((void**)&mInputDevice, srcW * srcH * img.channels() * sizeof(unsigned char)));
        CHECK(cudaMalloc((void**)&this->mInputResizeDevice, dstH * dstW * img.channels() * sizeof(float)));
        CHECK(cudaMalloc((void**)&this->mInputCHWDevice, dstH * dstW * img.channels() * sizeof(float)));

        // update d2s
        mScale = std::min(dstW / (img.cols*1.0), dstH / (img.rows*1.0));
        cv::Point2f srcPoints[3] = {
            cv::Point2f(0, 0),     // 左上角
            cv::Point2f(img.cols-1, 0),     // 右上角
            cv::Point2f(0, img.rows-1)      // 左下角
        };

        cv::Point2f dstPoints[3] = {
            cv::Point2f(0, 0),     // 左上角
            cv::Point2f(img.cols * mScale - 1 , 0),     // 右上角
            cv::Point2f(0, img.rows* mScale - 1)      // 左下角
        };
        cv::Mat M = cv::getAffineTransform(dstPoints, srcPoints);
        md2s.v00 = M.at<double>(0, 0);
        md2s.v01 = M.at<double>(0, 1);
        md2s.v02 = M.at<double>(0, 2);
        md2s.v10 = M.at<double>(1, 0);
        md2s.v11 = M.at<double>(1, 1);
        md2s.v12 = M.at<double>(1, 2);
    }
    CHECK(cudaMemcpyAsync(mInputDevice, img.data, srcW * srcH * img.channels() * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
    ResizePaddingDevice(mInputDevice, srcW, srcH, mInputResizeDevice, dstW, dstH, mScale, md2s, stream);
    HWC2CHWDevice(mInputResizeDevice, mInputCHWDevice, dstW, dstH, img.channels(), stream);
}

void Yolo::PostProcessDevice(std::vector<Object>& objects, const cudaStream_t& stream)
{
    // clear keepFlag and count
    CHECK(cudaMemset(mOutputObjectDevice, 0, (1 + gridStrideSize * mOutputWidth) * sizeof(float)));
    objects.clear();

    GenerateYoloProposalDevice(gridStridesDevice, gridStrideSize, mOutputSrcDevice, 
    mOutputObjectDevice,bboxConfThresh, NumClasses, stream);

    FastNMSDevice(mOutputObjectDevice, iouThresh, mOutputWidth, topK, stream);
    CHECK(cudaMemcpyAsync(mOutputObjectHost, mOutputObjectDevice, 
    (1 + gridStrideSize * mOutputWidth) * sizeof(float), cudaMemcpyDeviceToHost, stream));

    int count = mOutputObjectHost[0];
    // std::cout << "num of boxes before nms: " << count << std::endl;
    for (int i = 0; i < count; ++i)
    {
        float* pobject = mOutputObjectHost + 1 + i * mOutputWidth;
        if (pobject[6])
        {
            float x0 = pobject[0] / mScale;
            float y0 = pobject[1] / mScale;
            float x1 = (pobject[0] + pobject[2]) / mScale;
            float y1 = (pobject[1] + pobject[3]) / mScale;
            // clip 
            x0 = std::max(std::min(x0, (float)(srcW - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(srcH - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(srcW - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(srcH - 1)), 0.f);

            Object object;
            object.x = x0;
            object.y = y0;
            object.w = x1 - x0;
            object.h = y1 - y0;
            object.prob = pobject[4];
            object.label = pobject[5];
            objects.push_back(object);
        }
    }
    // std::cout << "num of boxes after nms: " << objects.size() << std::endl;
}

Yolo::Yolo(int dstH, int dstW, int NumClasses, float bboxConfThresh, float iouThresh, bool isDevice)
{
    this->NumClasses = NumClasses;
    this->dstH = dstH;
    this->dstW = dstW;
    this->mOutputWidth = 7;
    this->bboxConfThresh = bboxConfThresh;
    this->iouThresh = iouThresh;
    this->isDevice = isDevice;

    // generate grid strides
    this->gridStrideSize = 0; // fix out of memeory error 
    for (auto stride : this->strides)
    {
        int num_grid_y = dstH / stride;
        int num_grid_x = dstW / stride;
        this->gridStrideSize += num_grid_y * num_grid_x;
    }
    this->gridStridesHost = (int*)malloc(this->gridStrideSize * 3 * sizeof(int));
    this->GenerateGridsAndStride(this->strides, this->gridStridesHost);

    if (isDevice)
    {
        CHECK(cudaMalloc((void**)&this->gridStridesDevice, 
        this->gridStrideSize * this->strides.size() * sizeof(int)));
        CHECK(cudaMemcpy(this->gridStridesDevice, this->gridStridesHost, 
        this->gridStrideSize * this->strides.size() * sizeof(int), cudaMemcpyHostToDevice));
        // output
        CHECK(cudaMalloc((void**)&this->mOutputSrcDevice, this->gridStrideSize * (4 + 1 + NumClasses) * sizeof(float)));
        CHECK(cudaMalloc((void**)&this->mOutputObjectDevice, (1 + gridStrideSize * this->mOutputWidth) * sizeof(float)));
        CHECK(cudaMallocHost((void**)&this->mOutputObjectHost, (1 + gridStrideSize * this->mOutputWidth) * sizeof(float)));
    }
    else
    {
        this->mInputCHWHost = (float*)malloc(dstH * dstW * 3 * sizeof(float));
        this->mOutputSrcHost = (float*)malloc(this->gridStrideSize * (4 + 1 + NumClasses) * sizeof(float));
        this->mOutputObjectHost = (float*)malloc((1 + gridStrideSize * this->mOutputWidth) * sizeof(float));
    }
    
}

Yolo::~Yolo()
{
    std::cout << "===> Destroy Yolo instance" <<std::endl;
    if (this->isDevice)
    {
        CHECK(cudaFree(this->mInputResizeDevice));
        CHECK(cudaFree(this->mInputCHWDevice));
        CHECK(cudaFree(this->mInputDevice));
        CHECK(cudaFree(this->gridStridesDevice));
        CHECK(cudaFree(this->mOutputSrcDevice));
        CHECK(cudaFree(this->mOutputObjectDevice));

        CHECK(cudaFreeHost(this->mOutputObjectHost));
    }
    else{
        free(this->mOutputSrcHost);
        free(this->mInputCHWDevice);
        free(this->mOutputObjectHost);
    }
    free(this->gridStridesHost);   
}