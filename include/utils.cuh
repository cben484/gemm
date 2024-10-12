#ifndef UTILS_CUH
#define UTILS_CUH

#include "utils_check_fuctions.hpp"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <ostream>
#include <utility>

#define BLOCK_SIZE 32

template <typename T>
float compute_effective_bandwidth(size_t m, size_t n, size_t k, float latency) {
  return ((m * k + k * n + m * n) * sizeof(T)) / (latency * 1e-3) / 1e9;
}

float compute_effective_tflops(size_t m, size_t n, size_t k, float latency) {
  return (2.0 * m * k * n) / (latency * 1e-3) / 1e12;
}

void print_performance_result(size_t m, size_t n, size_t k, float latency) {
  float const effective_bandwidth{
      compute_effective_bandwidth<float>(m, n, k, latency)};
  float const effective_tflops{compute_effective_tflops(m, n, k, latency)};

  std::cout << "Latency: " << latency << " ms" << std::endl;
  std::cout << "Effective Bandwidth: " << effective_bandwidth << " GB/s"
            << std::endl;
  std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS"
            << std::endl;
}

template <typename T>
int elapsedTime(std::function<T(cudaStream_t)> bound_function,
                cudaStream_t stream, size_t num_warmups, size_t num_repeats,
                float latency);

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 1,
                          size_t num_warmups = 1) {
  return elapsedTime(bound_function, stream, num_warmups, num_repeats);
}

template <typename T>
int curandgenerate(T *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, seed));
  // CHECK_Curand(curandGenerateUniform(gen, matrx, Sum));
  curandSpecific(gen, matrx, Sum);
  CHECK_Curand(curandDestroyGenerator(gen));

  return EXIT_SUCCESS;
}
// 重载给curand使用
void curandSpecific(curandGenerator_t gen, float *matrx, size_t Sum) {
  CHECK_Curand(curandGenerateUniform(gen, matrx, Sum));
}
void curandSpecific(curandGenerator_t gen, double *matrx, size_t Sum) {
  CHECK_Curand(curandGenerateUniformDouble(gen, matrx, Sum));
}

template <typename T>
float elapsedTime(std::function<T(cudaStream_t)> bound_function,
                  cudaStream_t stream, size_t num_warmups, size_t num_repeats) {
  cudaEvent_t start, stop;
  float time;
  CHECK_Runtime(cudaEventCreate(&start));
  CHECK_Runtime(cudaEventCreate(&stop));

  for (size_t i{0}; i < num_warmups; ++i) {
    bound_function(stream);
  }

  CHECK_Runtime(cudaStreamSynchronize(stream));
  CHECK_Runtime(cudaEventRecord(start, stream));
  for (size_t i{0}; i < num_repeats; ++i) {
    bound_function(stream);
  }
  CHECK_Runtime(cudaEventRecord(stop, stream));
  CHECK_Runtime(cudaEventSynchronize(stop));
  CHECK_Runtime(cudaEventElapsedTime(&time, start, stop));
  CHECK_Runtime(cudaEventDestroy(start));
  CHECK_Runtime(cudaEventDestroy(stop));

  return time / num_repeats;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
constexpr cudaDataType_t cuda_data_type_trait() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, __half>::value) {
    return CUDA_R_16F;
  } else {
    throw std::runtime_error("Unsupported data type.");
  }
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
void benchmark(size_t m, size_t n, size_t k, T const *A, T const *B, T *C,
               cublasHandle_t handle, T alpha, T beta) {

  // std::cout << "实际上使用cublasDgemm两矩阵相乘的参数为：" << m << "x" << k
  //           << " 矩阵与 " << k << "x" << n << " 矩阵的乘法" << std::endl;

  // double alpha = 1.0;
  // double beta = 0.0;
  constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
  constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};

  int lda = k;
  int ldb = n;
  int ldc = n;

  // CHECK_Cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
  // A,
  //                          lda, B, ldb, &beta, C, ldc));
  CHECK_Cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                            B, data_type, ldb, A, data_type, lda, &beta, C,
                            data_type, ldc, data_type, algo));
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
std::pair<float, float>
lunch(size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc,
      std::function<void(size_t, size_t, size_t, T const *, size_t, T const *,
                         size_t, T *, size_t, const double, const double,
                         cudaStream_t)>
          gemm_kernel_launch_function) {

  std::cout << "实际上两矩阵相乘的参数为：" << m << "x" << k << " 矩阵与 " << k
            << "x" << n << " 矩阵的乘法" << std::endl;

  cudaStream_t stream;
  CHECK_Runtime(cudaStreamCreate(&stream));
  cublasHandle_t handle;
  CHECK_Cublas(cublasCreate(&handle));
  CHECK_Cublas(cublasSetStream(handle, stream));

  T const alpha{(1.0)};
  T const beta{(0.0)};

  T *A{nullptr};
  T *B{nullptr};
  T *C{nullptr};
  CHECK_Runtime(cudaMalloc((void **)&A, sizeof(T) * m * k));
  CHECK_Runtime(cudaMalloc((void **)&B, sizeof(T) * k * n));
  CHECK_Runtime(cudaMalloc((void **)&C, sizeof(T) * m * n));

  curandgenerate(A, m, k, 1234ULL);
  curandgenerate(B, m, k, 4321ULL);
  curandgenerate(C, m, k, 3214ULL);

  float const latency_cublas{measure_performance<void>(
      [&](cudaStream_t stream) {
        benchmark(m, n, k, A, B, C, handle, alpha, beta);
      },
      stream)};
  float const latency_kernel{measure_performance<void>(
      [&](cudaStream_t stream) {
        gemm_kernel_launch_function(m, n, k, A, k, B, n, C, n, alpha, beta,
                                    stream);
      },
      stream)};

  std::cout << "cublas performance：" << latency_cublas << " ms  "
            << latency_cublas * 1e-3 << " s" << std::endl;
  print_performance_result(m, n, k, latency_cublas);
  std::cout << "kernel performance：" << latency_kernel << " ms  "
            << latency_kernel * 1e-3 << " s" << std::endl;
  print_performance_result(m, n, k, latency_kernel);
  std::cout << "Custom GEMM VS cuBLAS GEMM Performance: "
            << latency_cublas / latency_kernel * 100.0f << "%" << std::endl;

  CHECK_Runtime(cudaFree(A));
  CHECK_Runtime(cudaFree(B));
  CHECK_Runtime(cudaFree(C));

  return std::pair<float, float>{latency_cublas, latency_kernel};
}

#endif