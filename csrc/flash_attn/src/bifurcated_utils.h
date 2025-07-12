/******************************************************************************
 * Copyright (c) 2024, AWS NGDE
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector Utils
// CUDA kernel to pause for at least num_cycle cycles
template<typename cycle_type>
__forceinline__ __device__ void sleep(cycle_type num_cycles)
{
  int64_t cycles = 0;
  int64_t start = clock64();
  while(cycles < num_cycles) {
    cycles = clock64() - start;
  }
}

// Print a tensor's value at certain indices (good for debugging comparing
// with the same tensor on the pytorch side).
template<typename Tensor0>
__forceinline__ __device__ void bifurcated_check_tensor(Tensor0 tensor, int n_block) {
  sleep((int64_t)(6 * 100000ULL));
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
    printf("\n============ bifurcated_check_tensor =========== n_block = %d\n", n_block);
    cute::print(tensor);
    printf("\n");
    printf("%3.4f, \n", static_cast<float>(tensor(make_coord(0, 0), 0, 0)));
    printf("%3.4f, \n", static_cast<float>(tensor(make_coord(1, 0), 0, 0)));
    printf("%3.4f, \n", static_cast<float>(tensor(make_coord(2, 0), 0, 0)));
    printf("%3.4f, \n", static_cast<float>(tensor(make_coord(3, 0), 0, 0)));
    printf("\n----------- bifurcated_check_tensor ------------ n_block = %d\n", n_block);
  }
  sleep((int64_t)(6 * 100000ULL));
}

template<typename Tensor0>
__forceinline__ __device__ void vector_print_tensor(Tensor0 tensor) {
  //TODO: assert rank of tensor is supported!
  sleep((int64_t)(6 * 100000ULL));
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 8 && blockIdx.z == 45) {
    for (int r = 0; r < size<0>(tensor) && r < 1; r++) {
      for (int c = 0; c < size<1>(tensor); c++) {
        auto x = tensor(make_coord(r, c));
        printf("%3.4f, ", static_cast<float>(x));
      }
      printf("\n");
    }
  }
  sleep((int64_t)(6 * 100000ULL));
}

// acc_o is ((2,2),1,8):((1,2),0,4)
template<typename Tensor0>
__forceinline__ __device__ void vector_print_tensor_acc_o(Tensor0 tensor) {
  //TODO: assert rank of tensor is supported!
  sleep((int64_t)(6 * 100000ULL));
  if (true) {
	  printf("\n[vector_print_tensor_acc_o] \n=========== BEGIN ===========:\ntensor info:\n");
    print(tensor);
		printf("\n");
    for (int z = 0; z < size<2>(tensor); z++) {
      for (int y = 0; y < size<1>(tensor); y++) {
          for (int c = 0; c < size<0,1>(tensor) && c < 10; c++) {
            for (int r = 0; r < size<0,0>(tensor) && r < 25; r++) {
                auto x = tensor(make_coord(r, c), y, z);
                printf("%3.4f, ", static_cast<float>(x));
            }
            printf("\n");
          }
      }
      printf("\n\n");
    }
	  printf("\n=========== END ==========\n[vector_print_tensor_acc_o]\n");
  }
  sleep((int64_t)(6 * 100000ULL));
}

template<typename LatexPrintableType>
__forceinline__ __device__ void vector_print_latex(LatexPrintableType x) {
  sleep((int64_t)(6 * 100000000ULL));
  if (threadIdx.x == 1 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 45) {
    print_latex(x);
  }
  sleep((int64_t)(6 * 100000000ULL));
}

template<typename CutePrintableType>
__forceinline__ __device__ void vector_cute_print(CutePrintableType x) {
  sleep((int64_t)(6 * 100000000ULL));
  if (threadIdx.x == 1 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
	  printf("\n[vector_cute_print] \n=========== BEGIN ===========:\n");
	  cute::print(x);
	  printf("\n[vector_cute_print] \n===========  END  ===========:\n");
  }
  sleep((int64_t)(6 * 100000000ULL));
}

}  // namespace flash
