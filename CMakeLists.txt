cmake_minimum_required(VERSION 3.0)

project(yolox LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)

file(GLOB CPPS 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/*.h
)
list(REMOVE_ITEM CPPS ${PROJECT_SOURCE_DIR}/src/yolox.cpp)
set(src_list ${CPPS})

# include_directories(${PROJECT_SOURCE_DIR}/tiny-tensorrt)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/usr/lib/aarch64-linux-gnu/include)
link_directories(/usr/lib/aarch64-linux-gnu/lib64)
# tensorrt
include_directories(/usr/src/tensorrt/include)
link_directories(/usr/src/tensorrt/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolox ${src_list})

# target_compile_options(yolox PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     --generate-code arch=compute_53,code=sm_53
#     --generate-code arch=compute_62,code=sm_62
#     --generate-code arch=compute_72,code=sm_72
#     --generate-code arch=compute_75,code=sm_75
#     --generate-code arch=compute_86,code=sm_86
#     >)
target_link_libraries(yolox nvinfer)
target_link_libraries(yolox cudart)
target_link_libraries(yolox ${OpenCV_LIBS})

