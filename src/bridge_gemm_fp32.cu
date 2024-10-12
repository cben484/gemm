#include "../include/gemm_template.cuh"
#include "../include/utils.cuh"
#include "../include/utils_benchmark.cuh"
#include "../include/utils_check_device.cuh"
#include "../include/utils_check_fuctions.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

int main(int argc, char *argv[]) {

  float SelapsedTime;
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[3]);
  int k = std::stoi(argv[2]);
  int lda = ((std::stoi(argv[2]) + 15) / 16) * 16;
  int ldb = ((std::stoi(argv[3]) + 15) / 16) * 16;
  int ldc = ((std::stoi(argv[3]) + 15) / 16) * 16;

  cudaStream_t stream;
  CHECK_Runtime(cudaStreamCreate(&stream));

  std::cout << "***************************************************************"
               "***************************************************************"
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
  std::cout << "cublas(double)的时间开销：" << SelapsedTime << "ms  "
            << SelapsedTime / 1000 << "s" << std::endl;

  std::cout << "cublas的TFLOPS："
            << (2.0 * m * n * k) / ((SelapsedTime * 1e-3) * 1e12) << " TFLOPS"
            << std::endl;

  std::cout << "***************************************************************"
               "***************************************************************"
            << std::endl;

  std::vector<std::pair<
      std::string,
      std::function<void(size_t, size_t, size_t, float const *, size_t,
                         float const *, size_t, float *, size_t, const double,
                         const double, cudaStream_t)>>> const
      gemm_kernel_launch_functions{
          {"Custom GEMM Kernel V00", launch_gemm_kernel_v00<float>},
      };

  for (auto const &gemm_kernel_launch_function : gemm_kernel_launch_functions) {
    std::cout << gemm_kernel_launch_function.first << std::endl;
    std::pair<float, float> const gemm_result{lunch<float>(
        m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second)};
    std::cout << std::endl;
  }

  // elapsedTime([&](cudaStream_t stream{gemm_naive}), stream, );

  return EXIT_SUCCESS;
}