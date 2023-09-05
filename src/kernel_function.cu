#include <math.h>
#include <device_launch_parameters.h>
#include "kernel_function.h"
#define BLOCK_SIZE 16

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
void GenerateYoloProposalKernel(int* gridStrides, int gridStrideSize, float* outputSrc, 
                            float* objects, float bboxConfThresh, int numClass)
{
    // gridStrides: [8400, 3]
    // outputSrc: [bs, 8400, 4 + 1 + NumClasses]
    // objects: 1 + [8400, 7] 1: save keepFlag count 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    int anchorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int classIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (anchorIdx  >= gridStrideSize || classIdx >= numClass) 
        return;
    int x = gridStrides[anchorIdx * 3];
    int y = gridStrides[anchorIdx * 3 + 1];
    int stride = gridStrides[anchorIdx * 3 + 2];
    float* psrc = outputSrc + anchorIdx * (4 + 1 + numClass);

    float box_objecness = psrc[4];
    float box_cls_score = psrc[5 + classIdx];
    float box_prob = box_objecness * box_cls_score;
    if (box_prob > bboxConfThresh)
    {
        int index = atomicAdd(objects, 1);
        float* pobject = objects + 1 + index * 7;

        float x0 = (x + psrc[0]) * stride;
        float y0 = (y + psrc[1]) * stride;
        float w = exp(psrc[2]) * stride;
        float h = exp(psrc[3]) * stride;
        // caculate left top 
        x0 = x0 - w * 0.5f;
        y0 = y0 - h * 0.5f;
        *pobject++ = x0;
        *pobject++ = y0;
        *pobject++ = w;
        *pobject++ = h;
        *pobject++ = box_prob;
        *pobject++ = classIdx;
        *pobject++ = 1;
    }
}

__device__
float bboxIouDevice(float ax1, float ay1, float aw, float ah,
              float bx1, float by1, float bw, float bh)
{
    float inter_x1 = max(ax1, bx1);
    float inter_y1 = max(ay1, by1);
    float inter_x2 = min(ax1 + aw, bx1 + bw);
    float inter_y2 = min(ay1 + ah, by1 + bh);

    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    float box1_area = aw * ah;
    float box2_area = bw * bh;
    float union_area = box1_area + box2_area - inter_area;
    if (union_area <= 0)
        return 0;
    return inter_area / union_area;
}

__global__
void FastNMS(float* objects, float iouThresh, int objectWidth, int topK)
{
    // objects 1 + [8400, 7] 1: count, 7: x0, y0, w, h, box_prob, class_idx, keepFlag
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = min(int(objects[0]), topK);
    if (idx >= objects[0])
        return;
    float* pcurrent = objects + 1 + idx * objectWidth;
    if (pcurrent[6] == 0)
        return;
    for (int i = 0; i < count; ++i)
    {
        float* pitem = objects + 1 + i * objectWidth;
        if (i == idx || pitem[5] != pcurrent[5])
            continue;
        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < idx)
                continue;
            float iou = bboxIouDevice(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                                      pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > iouThresh)
            {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

// (x, y) -> (projX, projY)
__device__
void AffineProjectKernel(AffineMatrix* matrix, int x, int y, float* projX, float* projY)
{
    *projX = matrix->v00 * x + matrix->v01 * y + matrix->v02;
    *projY = matrix->v10 * x + matrix->v11 * y + matrix->v12;
}

__global__
void ResizeNearestPaddingKernel(uchar* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, 
                                int widthPadding, int heightPadding, uchar paddingValue, AffineMatrix d2s)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // dstWidth
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // dstHeight
    if (dx >= dstWidth || dy >= dstHeight)
        return;
    float projX = 0;
    float projY = 0;
    AffineProjectKernel(&d2s, dx, dy, &projX, &projY);
    float c0, c1, c2;
    if (dx >= dstWidth - widthPadding || dy >= dstHeight - heightPadding)
    {
        c0 = paddingValue;
        c1 = paddingValue;
        c2 = paddingValue;
    }
    else
    {
        int srcX = floor(projX + 0.5f);
        int srcY = floor(projY + 0.5f);
        
        if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight)
        {
            c0 = paddingValue;
            c1 = paddingValue;
            c2 = paddingValue;
        }
        else
        {
            c0 = static_cast<float>(src[srcY * srcWidth * 3 + srcX * 3]);
            c1 = static_cast<float>(src[srcY * srcWidth * 3 + srcX * 3 + 1]);
            c2 = static_cast<float>(src[srcY * srcWidth * 3 + srcX * 3 + 2]);
        }
    }
    float* pdst = dst + dy * dstWidth * 3 + dx * 3;
    *pdst++ = c0;
    *pdst++ = c1;
    *pdst++ = c2;
}

__global__
void ResizeBilinearPaddingKernel(uchar* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight,
                                int widthPadding, int heightPadding, uchar paddingValue, AffineMatrix d2s)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // dstWidth
    int dy = blockIdx.y * blockIdx.y + threadIdx.y;  // dstHeight
    if (dx >= dstWidth || dy >= dstHeight)
        return;
    // coming soon
}
__global__
void HWC2CHWKernel(float* src, float* dst, int width, int height, int channels)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= width || h >= height)
        return;
    for (int c = 0; c < channels; ++c)
    {
        dst[c * height * width + h * width + w] = src[h * width * channels + w * channels + c];
    }
}

void FastNMSDevice(float* objects, float iouThresh, int objectWidth, int topK, const cudaStream_t& stream)
{
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((topK + blockSize.x - 1) / blockSize.x);
    FastNMS<<<gridSize, blockSize, 0, stream>>>(objects, iouThresh, objectWidth, topK);
}

void GenerateYoloProposalDevice(int* gridStrides, int gridStrideSize, float* outputSrc, float* objects,
                            float bboxConfThresh, int numClass, const cudaStream_t& stream)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((gridStrideSize + blockSize.x - 1) / blockSize.x, (numClass + blockSize.y - 1) / blockSize.y);
    GenerateYoloProposalKernel<<<gridSize, blockSize, 0, stream>>>(gridStrides, gridStrideSize, outputSrc, objects, bboxConfThresh, numClass);
}

void ResizePaddingDevice(uchar* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, float scale, AffineMatrix d2s, const cudaStream_t& stream)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    int heightPadding = (dstHeight - srcHeight * scale);
    int widthPadding = (dstWidth - srcWidth * scale);
    uchar paddingValue = 114;
    ResizeNearestPaddingKernel<<<gridSize, blockSize, 0, stream>>>(src, srcWidth, srcHeight, dst, dstWidth, dstHeight, widthPadding, heightPadding, paddingValue, d2s);
}

void HWC2CHWDevice(float* src, float* dst, int width, int height, int channels, const cudaStream_t& stream)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    HWC2CHWKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height, channels);
}

void blobFromImageCuda(float* blobDev, const cv::Mat& img, const cudaStream_t& stream) 
{
    int channels = img.channels();
    int img_h = img.rows;
    int img_w = img.cols;

    // Copy image data from CPU to GPU using the provided stream
    uchar* imgDataDev = nullptr;
    cudaMalloc((void**)&imgDataDev, img.total() * img.channels() * sizeof(uchar));
    cudaMemcpyAsync(imgDataDev, img.data, img.total() * img.channels() * sizeof(uchar), cudaMemcpyHostToDevice, stream);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((img_w + blockSize.x - 1) / blockSize.x, (img_h + blockSize.y - 1) / blockSize.y);

    blobFromImageKernel<<<gridSize, blockSize, 0, stream>>>(imgDataDev, blobDev, channels, img_h, img_w);

    cudaFree(imgDataDev);  // Free GPU memory used for image data
}


// int main()
// {
//     cv::Mat img = cv::imread("car.jpg");
//     int channels = img.channels();
//     int img_h = img.rows;
//     int img_w = img.cols;
//     int img_size = img.total() * img.channels();
//     uchar* imgDataDev = nullptr;
//     cudaMalloc((void**)&imgDataDev, img_size * sizeof(uchar));
//     cudaMemcpy(imgDataDev, img.data, img_size * sizeof(uchar), cudaMemcpyHostToDevice);

//     int dst_w = 640;
//     int dst_h = 640;

//     float* dstImage = nullptr;
//     cudaMalloc((void**)&dstImage, dst_w * dst_h * channels * sizeof(float));

//     float scale = std::min(dst_w / (img_w * 1.0f), dst_h / (img_h * 1.0f));

//     std::cout << scale << std::endl;

//         // 源图像上的三个点和目标图像上的对应三个点
//     cv::Point2f srcPoints[3] = {
//         cv::Point2f(0, 0),     // 左上角
//         cv::Point2f(img_w-1, 0),     // 右上角
//         cv::Point2f(0, img_h-1)      // 左下角
//     };

//     cv::Point2f dstPoints[3] = {
//         cv::Point2f(0, 0),     // 左上角
//         cv::Point2f((int) img_w * scale - 1 , 0),     // 右上角
//         cv::Point2f(0, (int) img_h* scale - 1)      // 左下角
//     };

//     cv::Mat M = cv::getAffineTransform(dstPoints, srcPoints);

//     std::cout << "Scale is: " << 1.0f / scale << std::endl;

//     std::cout << "Affine Transformation Matrix:\n" << M << std::endl;

    
//     std::cout<< "Affine Transformation Matrix:\n" << d2s.v00 << " " << d2s.v01 << " " << d2s.v02 << " " << d2s.v10 << " " << d2s.v11 << " " << d2s.v12 << std::endl;

//     AffineMatrix d2s;
//     d2s.v00 = M.at<double>(0, 0);
//     d2s.v01 = M.at<double>(0, 1);
//     d2s.v02 = M.at<double>(0, 2);
//     d2s.v10 = M.at<double>(1, 0);
//     d2s.v11 = M.at<double>(1, 1);
//     d2s.v12 = M.at<double>(1, 2);

    
//     ResizePaddingDevice(imgDataDev, img_w, img_h, dstImage, dst_w, dst_h, scale, d2s, 0);

//     float* hostData = new float[dst_w * dst_h * channels];
//     cudaMemcpy(hostData, dstImage, dst_w * dst_h * channels * sizeof(float), cudaMemcpyDeviceToHost);
//     cv::Mat result(dst_h, dst_w, CV_32FC(channels), hostData);

//     result.convertTo(result, CV_8UC3);
//     cv::imwrite("result.jpg", result);

//     delete[] hostData;
//     return 0;
// }