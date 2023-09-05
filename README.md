# TensorRT-YOLOXu

[![Cuda](https://img.shields.io/badge/CUDA-11.3-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)  [![TensorRT](./img/TensorRT.svg)](https://developer.nvidia.com/nvidia-tensorrt-8x-download) [![ubuntu](img/ubuntu.svg)](https://releases.ubuntu.com/18.04/)

- 🔥 Support TensorRT for YOLOX.
- 🚀 Easy, isolation the detection and inference.
- ⚡ Fast, preprocess and postprocess with CUDA kernel function.

## ONNX2TRT

Please use the official export script to export the ONNX file. Then use trtexec to convert the ONNX file to trt engine. 

We provide the [ONNX Model](https://drive.google.com/file/d/19k7AxSO0Sn84OLqCNOxBfZLp8mXUTPCh/view?usp=drive_link), please place in the workspace folder and unzip, convert into trtengine.

```sh
trtexec \
--onnx=yolox_s.onnx \
--saveEngine=yolox_s.trt \
--minShapes=images:1x3x640x640 \
--optShapes=images:1x3x640x640 \
--maxShapes=images:1x3x640x640 \
--memPoolSize=workspace:2048MiB
```

## Infer

🔥 We isolate detection and inference into two classes: 'Infer' and 'Yolo.

The 'Infer' class is designed to make learning TensorRT easier, especially when it comes to setting parameters, managing memory, and performing inference. 

You only need to load the TRT engine and prepare the memory.

```c++
// create infer
Infer* infer = new Infer(engine_dir, ILogger::Severity::kWARNING);
// memory copy 
infer->CopyFromHostToDevice(input, 0);
// inference
infer->Forward();
// memory copy 
infer->CopyFromDeviceToHost(output, 1);
```

## Yolo

The 'Yolo' class is designed for fast detection and deployment on edge devices. 

We perform all preprocessing and postprocessing using CUDA operations, achieving fast detection.

```c++
// create yolo 
Yolo* yolo = new Yolo(INPUT_H, INPUT_W, NUM_CLASSES, BBOX_CONF_THRESH, IOU_THRESH, USE_DEVICE);
// INPUT_H and INPUT_W is the shape of the input of engine

// prepare to save the objects
std::vector<Object> objects;

// load image
cv::Mat img = cv::imread(input_image_path);
// preprocess and inference
yolo->PreProcess(img);
infer->CopyFromHostToDevice(yolo->mInputCHWHost, 0);
infer->Forward();
infer->CopyFromDeviceToHost(yolo->mOutputSrcHost, 1);
yolo->PostProcess(objects);
```

## Reference

1. [TensorRT-Alpha](https://github.com/FeiYull/TensorRT-Alpha)
2. [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt)
3. [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)