#include <iostream>
#include <string>
#include "infer.h"
#include "utils.h"
#include "kernel_function.h"
#include "yolo.h"

using namespace nvinfer1;

#define USE_DEVICE true
#define DEVICE 0  // GPU id
#define IOU_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

#define  INPUT_W 640
#define  INPUT_H 640
#define  NUM_CLASSES 80


int main() 
{
    if (USE_DEVICE)
        cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    std::string engine_dir = std::string("./yolox_s_int8.trt");
    const std::string input_image_path = std::string("./car.jpg");

    Infer* infer = new Infer(engine_dir, ILogger::Severity::kWARNING);
    Yolo* yolo = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, IOU_THRESH, USE_DEVICE);
    
    // read and resize image
    cv::Mat img = cv::imread(input_image_path);
    std::vector<Object> objects;

    // // CPU
    // yolo->PreProcess(img);
    // // warm up
    // std::cout << "===>Warm up!" << std::endl;
    // for (int i = 0; i < 100; i++)
    // {
    //     yolo->PreProcess(img);
    //     infer->CopyFromHostToDevice(yolo->mInputCHWHost, 0);
    //     infer->Forward();
    //     infer->CopyFromDeviceToHost(yolo->mOutputSrcHost, 1);
    //     yolo->PostProcess(objects);
    // }

    // // run inference
    // std::cout << "===>Do inference!" << std::endl;
    // double start = get_time();
    // int ntest = 1000;
    // for (int i = 0; i < ntest; i++)
    // {
    //     yolo->PreProcess(img);
    //     infer->CopyFromHostToDevice(yolo->mInputCHWHost, 0);
    //     infer->Forward();
    //     infer->CopyFromDeviceToHost(yolo->mOutputSrcHost, 1);
    //     yolo->PostProcess(objects);
    // }
    // double end = get_time();
    // draw_objects_save(img, objects, input_image_path);
    // float avgInferTime = (end - start) / ntest;
    // std::cout << "===>Inference time: " << avgInferTime << "ms, " << "FPS: " << 1000 / avgInferTime << std::endl;

    // // GPU 
    // // warm up
    // std::cout << "===> Warm up!" << std::endl;
    // for (int i = 0; i < 100; i++)
    // {
    //     yolo->PreProcessDevice(img);
    //     infer->CopyFromDeviceToDeviceIn(yolo->mInputCHWDevice, 0);
    //     infer->Forward();
    //     infer->CopyFromDeviceToDeviceOut(yolo->mOutputSrcDevice, 1);
    //     yolo->PostProcessDevice(objects);
    // }
    
    // // run inference
    // std::cout << "===> Do inference!" << std::endl;
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    // double start = get_time();
    // int ntest = 1000;

    // for (int i = 0; i < ntest; i++)
    // {
    //     yolo->PreProcessDevice(img, stream);
    //     infer->CopyFromDeviceToDeviceIn(yolo->mInputCHWDevice, 0, stream);
    //     infer->Forward(stream);
    //     infer->CopyFromDeviceToDeviceOut(yolo->mOutputSrcDevice, 1, stream);
    //     yolo->PostProcessDevice(objects, stream);
    // }
    // double end = get_time();
    // cudaStreamDestroy(stream);

    // draw_objects_save(img, objects, input_image_path);

    // float avgInferTime = (end - start) / ntest;
    // std::cout << "===> Inference time: " << avgInferTime << "ms, " << "FPS: " << 1000 / avgInferTime << std::endl;
    
    delete infer;
    delete yolo;

    return 0;
}
