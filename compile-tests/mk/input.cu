#include <cuda.h>
#include "test_include.cuh"

#include <iostream>

__device__ void devicefun() {

}

extern "C" __global__ void testKernel() {
  printf("Hello from %d\n", threadIdx.x);
}

int main(void) {
  std::cout << "Hello World" << std::endl;
  testKernel<<<1,10>>>();
  cudaDeviceSynchronize();
  return 0;
}
