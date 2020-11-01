#ifndef KERMA_RT_DEVICE_INSTRUMENTATION_H
#define KERMA_RT_DEVICE_INSTRUMENTATION_H

__device__ void __record_load(unsigned int blockIdx, unsigned int threadIdx, unsigned int location) {
  printf("Block %d: tid %d: load %d\n", blockIdx, threadIdx, location);
}

__device__ void __record_store(unsigned int blockIdx, unsigned int threadIdx, unsigned int location) {
  printf("Block %d: tid %d: store %d\n", blockIdx, threadIdx, location);
} // namespace kerma

__device__ void __kerma_test_device_call() {
  printf("Hello from kerma\n");
}

#endif