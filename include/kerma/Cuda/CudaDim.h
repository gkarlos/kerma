#ifndef KERMA_CUDA_CUDA_DIM_H
#define KERMA_CUDA_CUDA_DIM_H

#include <kerma/Cuda/Cuda.h>

namespace kerma
{

class CudaDim
{
public:
  CudaDim();
  CudaDim(unsigned int x,
          unsigned int y,
          unsigned int z);
  CudaDim(const CudaDim &other);

  unsigned int x;
  unsigned int y;
  unsigned int z;

  void operator=(CudaDim &other);
  bool operator==(CudaDim &other);
};

enum CudaDimError : int
{
  UnknownCompute = -6, 
  InvalidDimX,
  InvalidDimY,
  InvalidDimZ,
  UnsupportedDimZ,
  Failure,
  Success
};

CudaDimError validateGrid(CudaDim &dim);

CudaDimError validateGrid(CudaCompute &compute, CudaDim &dim);

CudaDimError validateBlock(CudaDim &dim);

CudaDimError validateBlock(CudaCompute &compute, CudaDim &dim);


} /* NAMESPACE kerma */


#endif