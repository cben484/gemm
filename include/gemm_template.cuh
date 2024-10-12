#ifndef GEMM_TEMPLATE_CUH
#define GEMM_TEMPLATE_CUH

#include <cuda_runtime.h>

template<typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const *A, size_t lda,
                           T const *B, size_t ldb, T *C, size_t ldc,
                           const double alpha, const double beta,
                           cudaStream_t stream);

#endif