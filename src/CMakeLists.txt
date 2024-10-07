cmake_minimum_required(VERSION 3.25)

project(GEMM VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS utils_check_fuctions.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB CUDA_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

add_library(00_gemm_naive SHARED 00_gemm_naive.cu)

target_include_directories(00_gemm_naive PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# c++版本检查
add_executable(version_exec checktheversion.cpp)

# 编译出gemm的benchmark的exe
add_executable(00_gemm_naive_exec 00_gemm_naive.cu)
target_link_libraries(00_gemm_naive_exec 00_gemm_naive CUDA::curand CUDA::cublas CUDA::cusolver)
set_target_properties(00_gemm_naive PROPERTIES CUDA_ARCHITECTURES native)