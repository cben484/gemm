#include "../include/utils_benchmark.cuh"
#include "../include/utils_check_device.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {

  float SelapsedTime;
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[3]);
  int k = std::stoi(argv[2]);

  std::cout << "***************************************************************"
               "***************************************************************"
               "*************************************"
            << std::endl;
  std::cout << "探测设备......" << std::endl;
  CHECK_Device(&argv[0]);

  std::cout << "参数总数：" << std::endl;
  std::cout << argc << std::endl;
  std::cout << "参数检查：" << std::endl;
  std::cout << argv[0] << std::endl;
  std::cout << argv[1] << std::endl;
  std::cout << argv[2] << std::endl;
  std::cout << argv[3] << std::endl;
  std::cout << "参数检查完毕" << std::endl;

  std::cout << "实际上两矩阵相乘的参数为：" << argv[1] << "x" << argv[2]
            << " 矩阵与 " << argv[2] << "x" << argv[3] << " 矩阵相乘"
            << std::endl;

  benchmark(argc, argv, &SelapsedTime);
  std::cout << "cublas的时间开销：" << SelapsedTime << "ms  "
            << SelapsedTime / 1000 << "s" << std::endl;

  std::cout << "cublas的TFLOPS："
            << (2.0 * m * n * k) / ((SelapsedTime * 1e-3) * 1e12) << " TFLOPS"
            << std::endl;

  std::cout << "***************************************************************"
               "***************************************************************"
               "*************************************"
            << std::endl;

  return EXIT_SUCCESS;
}

template <typename T>
__global__ void gemm_naive(int m, int n, int k, T const *A, size_t lda,
                           T const *B, size_t ldb, T const *C, size_t ldc,
                           T alpha, T beta, float *time) {

  size_t row{blockDim.x * blockIdx.x + threadIdx.x};
  size_t col{blockDim.x * blockIdx.x + threadIdx.x};

  if (row < m && col < n) {
    double sum{0};
    for (size_t t{0}; t < k; t++) {
      sum += A[row * lda + t] * B[ldb * t + col];
    }
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
  }
}