#include "stdio.h"

__global__ void add(int a, int b, int *c)
{
    int val = a + 5;
    c[threadIdx.x] = val;
}

int main()
{
    int a, b, c, blockSz;
    int *dev_c;
    a=3;
    b=4;
    blockSz = 10;
    cudaMalloc((void**)&dev_c, sizeof(int) * blockSz);
    add<<<1, blockSz>>>(a, b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d is %d\n", a, b, c);
    cudaFree(dev_c);
    return 0;
}