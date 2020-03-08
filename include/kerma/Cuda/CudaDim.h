#ifndef KERMA_CUDA_CUDA_DIM_H
#define KERMA_CUDA_CUDA_DIM_H

namespace kerma
{

class CudaDim
{
public:
  CudaDim();
  CudaDim(unsigned int x,
          unsigned int y,
          unsigned int z);
  unsigned int x;
  unsigned int y;
  unsigned int z;

  void operator=(CudaDim &other);
  bool operator==(CudaDim &other);
};

int validateGrid(CudaDim &dim);

int validateBlock(CudaDim &dim);


} /* NAMESPACE kerma */


#endif