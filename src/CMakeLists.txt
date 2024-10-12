cmake_minimum_required(VERSION 3.25)

project(GEMM VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 --ptxas-options=-v")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS utils_check_fuctions.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB CUDA_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

add_library(cuda_gemm SHARED 00_gemm_naive.cu)
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)

# c++版本检查
add_executable(version_exec checktheversion.cpp)

# 入口exe
add_executable(bridge_fp64_exe bridge_gemm_fp32.cu)
target_link_libraries(bridge_fp64_exe cuda_gemm CUDA::cublas CUDA::curand CUDA::cusolver)
set_target_properties(bridge_fp64_exe PROPERTIES CUDA_ARCHITECTURES native)

# 编译出gemm的benchmark的exe
# add_executable(00_gemm_naive_exec 00_gemm_naive.cu)
# target_link_libraries(00_gemm_naive_exec 00_gemm_naive CUDA::curand CUDA::cublas CUDA::cusolver)
# set_target_properties(00_gemm_naive PROPERTIES CUDA_ARCHITECTURES native)