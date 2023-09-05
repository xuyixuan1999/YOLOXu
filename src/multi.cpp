#include <iostream>
#include <string>
#include "infer.h"
#include "utils.h"
#include "kernel_function.h"
#include "yolo.h"

using namespace nvinfer1;

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

#define  INPUT_W 640
#define  INPUT_H 640
#define  NUM_CLASSES 80

int main() 
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    std::string engine_dir = std::string("./yolox_s_int8.trt");
    const std::string input_image_path = std::string("./street.jpg");

    Infer* infer1 = new Infer(engine_dir, ILogger::Severity::kWARNING);
    Yolo* yolo1 = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, NMS_THRESH, true);
    Infer* infer2 = new Infer(engine_dir, ILogger::Severity::kWARNING);
    Yolo* yolo2 = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, NMS_THRESH, true);
    Infer* infer3 = new Infer(engine_dir, ILogger::Severity::kWARNING);
    Yolo* yolo3 = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, NMS_THRESH, true);
    
    // read and resize image
    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    cv::Mat pr_img = static_resize(img, INPUT_W, INPUT_H);
    std::cout << "blob image" << std::endl;

    // // allocate memory for output
    // std::vector<float> prob;

    // // blob in cuda
    float* blob = nullptr;
    int64_t blob_size = pr_img.total()*pr_img.channels();
    cudaMalloc((void**)&blob, blob_size * sizeof(float));
    blobFromImageCuda(blob, pr_img);
    infer1->CopyFromDeviceToDeviceIn(blob, 0);
    infer2->CopyFromDeviceToDeviceIn(blob, 0);
    infer3->CopyFromDeviceToDeviceIn(blob, 0);

    std::vector<Object> objects;
    
    // // warm up
    std::cout << "warm up" << std::endl;
    for (int i = 0; i < 100; i++)
    {
        infer1->Forward();
        infer3->Forward();
        infer3->Forward();
    }

    // create 3 stream
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    int ntest = 10000;
    double start = get_time(TimeUnit::SECONDS);
    double endTime = get_time(TimeUnit::SECONDS);
    while (endTime - start < 1200.0)
    {
        infer1->CopyFromDeviceToDeviceIn(blob, 0, stream1);
        infer2->CopyFromDeviceToDeviceIn(blob, 0, stream2);
        infer3->CopyFromDeviceToDeviceIn(blob, 0, stream3);
        infer1->Forward(stream1);
        infer2->Forward(stream2);
        infer3->Forward(stream3);
        infer1->CopyFromDeviceToDeviceOut(yolo1->tmpOutputSrc, 1, stream1);
        infer2->CopyFromDeviceToDeviceOut(yolo2->tmpOutputSrc, 1, stream2);
        infer3->CopyFromDeviceToDeviceOut(yolo3->tmpOutputSrc, 1, stream3);
        yolo1->DecodeOutputDevice(objects, scale, img_w, img_h, stream1);
        yolo2->DecodeOutputDevice(objects, scale, img_w, img_h, stream2);
        yolo3->DecodeOutputDevice(objects, scale, img_w, img_h, stream3);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);
        endTime = get_time(TimeUnit::SECONDS);
    }
    // double end = get_time();
    // std::cout << (end - start) / ntest << "ms" << std::endl;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    // draw_objects_save(img, objects, input_image_path);

    delete infer1;
    delete yolo1;
    delete infer2;
    delete yolo2;
    delete infer3;
    delete yolo3;

    return 0;
}
