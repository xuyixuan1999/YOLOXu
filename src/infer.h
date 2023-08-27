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
    void CopyFromHostToBufferH(const std::vector<float>& input, int bindIndex);
    void ChangeBufferD(int bindIndex, float* input, const cudaStream_t& stream = 0);
    void CopyFromHostToDevice(int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromDeviceToHost(int bindIndex, const cudaStream_t& stream = 0);
    void CopyFromBufferHToHost(std::vector<float>& output, int bindIndex);
    bool Forward();
    bool Forward(const cudaStream_t& stream);
    

private:
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    std::vector<std::string> mBindingName;

    std::vector<void*> cpu_buffer;

    std::vector<void*> gpu_buffer;

    bool mIsDynamicShape = false;

public:
    Infer(const std::string& engine_dir, nvinfer1::ILogger::Severity severity);
    ~Infer();
    
};


