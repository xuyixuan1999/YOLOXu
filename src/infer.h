#pragma once
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <vector>
#include <string>

class TrtLogger : public nvinfer1::ILogger
{
public:
    void setLogSeverity(Severity severity = Severity::kINFO);

private:
    Severity mSeverity;
    
    void log(Severity severity, const char* msg) noexcept override; 

};

class Infer
{
public:
    // inference
    void CopyFromDeviceToDeviceIn(float* input, int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromDeviceToDeviceOut(float* output, int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromHostToDevice(float* input, int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromDeviceToHost(float* output, int bindIndex, const cudaStream_t& stream = 0);
    bool Forward();
    bool Forward(const cudaStream_t& stream);

private:
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // params for engine
    std::vector<void*> mBinding;
    std::vector<size_t> mBindingSize;
    std::vector<nvinfer1::Dims> mBindingDims;
    std::vector<nvinfer1::DataType> mBindingDataType;
    std::vector<std::string> mBindingName;
    bool mIsDynamicShape = false;


public:
    Infer(const std::string& engine_dir, nvinfer1::ILogger::Severity severity);
    ~Infer();
    
};


