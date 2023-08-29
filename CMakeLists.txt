cmake_minimum_required(VERSION 3.0)

project(yolox LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

# please change the path of cuda cudnn and tensorrt to your own path
set(CUDA_DIR "/usr/local/cuda")
set(CUDNN_DIR "/usr/lib/aarch64-linux-gnu")
set(TENSORRT_DIR "/usr/src/tensorrt")

# add compute capability of the GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87")

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)

file(GLOB CPPS 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/*.h
)
list(REMOVE_ITEM CPPS ${PROJECT_SOURCE_DIR}/src/multi.cpp)
set(src_list ${CPPS})

# cuda
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)
# cudnn
include_directories(${CUDNN_DIR}/include)
link_directories(${CUDNN_DIR}/lib64)
# tensorrt
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolox ${src_list})

target_link_libraries(yolox nvinfer)
target_link_libraries(yolox cudart)
target_link_libraries(yolox ${OpenCV_LIBS})

