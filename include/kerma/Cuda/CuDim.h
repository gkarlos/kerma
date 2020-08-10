#ifndef KERMA_CUDA_CUDA_DIM_H
#define KERMA_CUDA_CUDA_DIM_H

#include <kerma/Cuda/Cuda.h>

namespace kerma
{

/// Represent Cuda dimension info
/// Similar to a dim3 but with extended functionality
class CuDim
{
public:
  CuDim()=delete;
  
  /**
   * @brief Construct a new CudaDim object
   * 
   * @param x Value of the x-dimension
   * @param y Value of the y-dimension
   * @param z Value of the z-dimension
   */
  CuDim(unsigned int x=1, unsigned int y=1,unsigned int z=1);
  CuDim(const CuDim &other);

  unsigned int x;
  unsigned int y;
  unsigned int z;

  void operator=(const CuDim &other);
  bool operator==(const CuDim &other) const;

public:
  unsigned int xMin();
  unsigned int xMax();
  unsigned int yMin();
  unsigned int yMax();
  unsigned int zMin();
  unsigned int zMax();
  unsigned int size();
  bool is1D();
  bool is2D();
  bool is3D();
  bool hasIdx();
};

// enum CudaDimError : int
// {
//   UnknownCompute = -6, 
//   InvalidDimX,
//   InvalidDimY,
//   InvalidDimZ,
//   UnsupportedDimZ,
//   Failure,
//   Success
// };

// CudaDimError validateGrid(CudaDim &dim);

// /*!
//  * @brief Check if the grid dimensions are "correct" w.r.t some Compute Capability
//  * @param compute [in] a CudaCompute
//  * @param dim [in] a CudaDim for the grid
//  * @return CudaDimError indicating status
//  */
// CudaDimError validateGrid(CudaCompute compute, CudaDim &dim);

// CudaDimError validateBlock(CudaDim &dim);

// /*!
//  * @brief Check if the block dimensions are "correct" w.r.t some Compute Capability
//  * @param compute [in] a CudaCompute
//  * @param dim [in] a CudaDim for the blocks
//  * @return CudaDimError indicating status
//  */
// CudaDimError validateBlock(CudaCompute compute, CudaDim &dim);


} /* NAMESPACE kerma */


#endif