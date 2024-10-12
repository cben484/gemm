#include "../include/utils.cuh"
#include "../include/utils_benchmark.cuh"
#include "../include/utils_check_device.cuh"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

template <typename T>
__global__ void gemm_00_naive(size_t m, size_t n, size_t k, T const *A,
                              size_t lda, T const *B, size_t ldb, T *C,
                              size_t ldc, const double alpha,
                              const double beta) {

  size_t row{blockDim.x * blockIdx.x + threadIdx.x};
  size_t col{blockDim.y * blockIdx.y + threadIdx.y};

  if (row < m && col < n) {
    double sum{0};
    for (size_t t{0}; t < k; t++) {
      sum += A[row * lda + t] * B[ldb * t + col];
    }
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
  }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const *A,
                            size_t lda, T const *B, size_t ldb, T *C,
                            size_t ldc, const double alpha, const double beta,
                            cudaStream_t stream) {
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(((m) + dimBlock.x - 1) / dimBlock.x,
               ((n) + dimBlock.y - 1) / dimBlock.y);
  gemm_00_naive<T><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, A, lda, B, ldb, C,
                                                     ldc, alpha, beta);
}

template void
launch_gemm_kernel_v00<float>(size_t m, size_t n, size_t k, float const *A,
                              size_t lda, float const *B, size_t ldb, float *C,
                              size_t ldc, const double alpha, const double beta,
                              cudaStream_t stream);

template void
launch_gemm_kernel_v00<double>(size_t m, size_t n, size_t k, double const *A,
                               size_t lda, double const *B, size_t ldb,
                               double *C, size_t ldc, const double alpha,
                               const double beta, cudaStream_t stream);