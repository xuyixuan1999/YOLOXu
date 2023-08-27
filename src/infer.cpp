#pragma once
#include "infer.h"
#include "error.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}

void TrtLogger::setLogSeverity(Severity severity) {
    this->mSeverity = severity;
}

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity > this->mSeverity)
    {
        return;
    }
    switch (severity)
    {
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERRROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    std::cerr << msg << std::endl;
}

void Infer::CopyFromHostToBufferH(const std::vector<float>& input, int bindIndex)
{
    memcpy(cpu_buffer[bindIndex], input.data(), mBindingSize[bindIndex]);
}

void Infer::CopyFromHostToDevice(int bindIndex, const cudaStream_t& stream)
{
    CHECK(cudaMemcpyAsync(gpu_buffer[bindIndex], cpu_buffer[bindIndex], 
    mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
}

void Infer::CopyFromDeviceToHost(int bindIndex, const cudaStream_t& stream)
{
    CHECK(cudaMemcpyAsync(cpu_buffer[bindIndex], gpu_buffer[bindIndex],
    mBindingSize[bindIndex], cudaMemcpyDeviceToDevice, stream));
}

void Infer::CopyFromBufferHToHost(std::vector<float>& output, int bindIndex)
{
    output.resize(mBindingSize[bindIndex]);
    memcpy(output.data(), cpu_buffer[bindIndex], mBindingSize[bindIndex]);
}

bool Infer::Forward()
{
    return mContext->executeV2(&gpu_buffer[0]);
}

bool Infer::Forward(const cudaStream_t& stream)
{
    return mContext->enqueueV2(gpu_buffer.data(), stream, nullptr);
}

void Infer::ChangeBufferD(int bindIndex, float* input, const cudaStream_t& stream)
{
    // void* blobVoidPtr = static_cast<void*>(input);
    CHECK(cudaMemcpyAsync(gpu_buffer[bindIndex], input, mBindingSize[bindIndex], cudaMemcpyDeviceToDevice, stream));
}

Infer::Infer(const std::string& engine_dir, nvinfer1::ILogger::Severity severity)
{  
    // create Logger
    TrtLogger gLogger;
    gLogger.setLogSeverity(severity);

    // load engine
    std::ifstream engine_file(engine_dir, std::ios::binary);
    long int fsize = 0;
    engine_file.seekg(0, engine_file.end);
    fsize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    std::vector<char> engine_string(fsize);
    engine_file.read(engine_string.data(), fsize);
    engine_file.close();
    if (engine_string.size() == 0)
    {
        std::cout << "Failed getting serialized engine!" << std::endl;
        return;
    }
    std::cout << "Succeeded getting serialized engine!" << std::endl;

    // deserialize engine
    mRuntime = createInferRuntime(gLogger);
    mEngine = mRuntime->deserializeCudaEngine(engine_string.data(), fsize);

    if (mEngine == nullptr)
    {
        std::cout << "Faild loading engine!" << std::endl;
        return;
    }
    std::cout << "Succeeded loading engine" << std::endl;

    // processing the bindings
    mContext = mEngine->createExecutionContext();
    int nbBindings = mEngine->getNbBindings();
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    cpu_buffer.resize(nbBindings);
    gpu_buffer.resize(nbBindings);
    
    for (int i = 0; i < nbBindings; i++)
    {
        // get binding info
        const char* name = mEngine->getBindingName(i);
        DataType dtype = mEngine->getBindingDataType(i);
        Dims dims = mEngine->getBindingDimensions(i);
        int dim_size = 1;
        for (int j = 0; j < dims.nbDims; j++)
        {
            dim_size *= abs(dims.d[j]);
        }
        int64_t totalSize = dim_size * getElementSize(dtype);
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;

        // print binding info
        if(mEngine->bindingIsInput(i)) {
            std::cout<< "input: ";
        } else {
            std::cout<< "output: ";
        }
        std::cout<<"binding bindIndex: "<< i << ", name: " << name<< ", size in byte: "<<totalSize;
        std::cout<<" binding dims with " << dims.nbDims << " dimemsion" << std::endl;;
    
        for(int j=0;j<dims.nbDims;j++) {
            std::cout << abs(dims.d[j]) << " x ";
        }
        std::cout << "\b\b  "<< std::endl;

        // allocate_buffers
        cpu_buffer[i] = nullptr;
        gpu_buffer[i] = nullptr;
        CHECK(cudaMallocHost(&cpu_buffer[i], totalSize));
        CHECK(cudaMalloc(&gpu_buffer[i], totalSize));
    }

}

Infer::~Infer()
{
    std::cout << "===> Destroy Infer instance" <<std::endl;
    for (size_t i = 0; i < this->mBinding.size(); i++)
    {
        CHECK(cudaFree(this->mBinding[i]));
    }
    for (size_t i = 0; i < this->gpu_buffer.size(); ++i)
    {
        cudaFreeHost(this->cpu_buffer[i]);
        cudaFree(this->gpu_buffer[i]);
    }
}