#ifndef UTILS_BENCHMARK_CUH
#define UTILS_BENCHMARK_CUH

#include "utils_check_device.cuh"
#include "utils_check_fuctions.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

int curandDgenerate(double *matrx, int m, int n, unsigned long long seed);

int benchmark(int argc, char *argv[], float *SelapsedTime) {

  std::cout << "benchmark接收参数" << std::endl;
  std::cout << argv[1] << std::endl;
  std::cout << argv[2] << std::endl;
  std::cout << argv[3] << std::endl;
  std::cout << "参数接收完毕" << std::endl;
  std::cout << "***************************************************************"
               "***************************************************************"
               "*************************************"
            << std::endl;
  std::cout << "实际上使用cublasDgemm两矩阵相乘的参数为：" << argv[1] << "x"
            << argv[2] << " 矩阵与 " << argv[2] << "x" << argv[3]
            << " 矩阵的乘法" << std::endl;

  cublasHandle_t handle;
  CHECK_Cublas(cublasCreate(&handle));
  double alpha = 1.0;
  double beta = 0.0;
  double *A, *B, *C;
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[3]);
  int k = std::stoi(argv[2]);
  int lda = std::stoi(argv[1]);
  int ldb = std::stoi(argv[2]);
  int ldc = std::stoi(argv[1]);
  CHECK_Runtime(cudaMalloc((void **)&A, sizeof(double) * m * k));
  CHECK_Runtime(cudaMalloc((void **)&B, sizeof(double) * k * n));
  CHECK_Runtime(cudaMalloc((void **)&C, sizeof(double) * m * n));

  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess) {
    printf("Failed to create start event\n");
    return EXIT_SUCCESS;
  }

  if (cudaEventCreate(&stop) != cudaSuccess) {
    printf("Failed to create stop event\n");
    CHECK_Runtime(cudaEventDestroy(start));
    return EXIT_SUCCESS;
  }

  curandDgenerate(A, m, k, 1234ULL);
  curandDgenerate(B, k, n, 4321ULL);

  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A,
                           lda, B, ldb, &beta, C, ldc));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));

  CHECK_Runtime(cudaEventElapsedTime(SelapsedTime, start, stop));

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return EXIT_SUCCESS;
}

int curandDgenerate(double *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, matrx, Sum);

  return EXIT_SUCCESS;
}

#endif