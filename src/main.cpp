#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <string>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "error.cuh"
#include "infer.h"
#include "utils.h"
#include "kernel_function.h"

using namespace nvinfer1;

int main() 
{
    // cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    std::string engine_dir = std::string("./yolox_s.trt");
    const std::string input_image_path = std::string("./car.jpg");

    Infer* infer = new Infer(engine_dir, ILogger::Severity::kWARNING);
    // prepare input data 
    std::vector<float> prob;

    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    cv::Mat pr_img = static_resize(img);
    std::cout << "blob image" << std::endl;



    float* blob = blobFromImageCuda(pr_img, 0);
    // infer->ChangeBufferD(0, blob);

    // float* blob;
    // blob = blobFromImage(pr_img);
    // size_t blob_size = pr_img.total()*3;
    // std::vector<float> input(blob, blob + blob_size);
    // // allocate_buffers
    // infer->CopyFromHostToBufferH(input, 0);
    // infer->CopyFromHostToDevice(0);
    // warm up
    std::cout << "warm up" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        infer->Forward();
    }

    // run inference
    double start = get_time();
    // infer->CopyFromHostToBufferH(input, 0);
    // infer->CopyFromHostToDevice(0);
    infer->ChangeBufferD(0, blob);
    for (int i = 0; i < 1000; i++)
    {
        infer->Forward();
    }
    infer->CopyFromDeviceToHost(1);
    infer->CopyFromBufferHToHost(prob, 1);
    std::vector<Object> objects;
    decode_outputs(prob.data(), objects, scale, img_w, img_h);
    draw_objects(img, objects, input_image_path);
    double end = get_time();
    std::cout << (end - start) * 1000 / 1000 << "ms" << std::endl;

    delete infer;

    return 0;
}
