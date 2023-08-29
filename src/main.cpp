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
    // cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    std::string engine_dir = std::string("./yolox_s.trt");
    const std::string input_image_path = std::string("./car.jpg");

    Infer* infer = new Infer(engine_dir, ILogger::Severity::kWARNING);
    Yolo* yolo = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, NMS_THRESH, false);
    
    // read and resize image
    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    cv::Mat pr_img = static_resize(img, INPUT_W, INPUT_H);
    std::cout << "blob image" << std::endl;

    // allocate memory for output
    std::vector<float> prob;

    // blob in cuda
    float* blob = nullptr;
    int64_t blob_size = pr_img.total()*pr_img.channels();
    cudaMalloc((void**)&blob, blob_size * sizeof(float));
    blobFromImageCuda(blob, pr_img);
    infer->CopyFromDeviceToDeviceIn(blob, 0);
    
    // blob in cpu
    // float* blob;
    // blob = blobFromImage(pr_img);

    // warm up
    std::cout << "warm up" << std::endl;
    for (int i = 0; i < 100; i++)
    {
        infer->Forward();
    }

    // run inference
    std::vector<Object> objects;
    double start = get_time();
    infer->CopyFromDeviceToDeviceIn(blob, 0);
    for (int i = 0; i < 1000; i++)
    {
        infer->Forward();
        infer->CopyFromDeviceToHost(prob, 1);
        cudaDeviceSynchronize();
    }
    double end = get_time();
    std::cout << (end - start) * 1000 / 1000 << "ms" << std::endl;

    objects = yolo->DecodeOutput(prob.data(), scale, img_w, img_h);
    draw_objects(img, objects, input_image_path);

    delete infer;
    delete yolo;

    return 0;
}
