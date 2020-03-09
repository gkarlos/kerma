#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/CudaDim.h>
#include <kerma/Support/Util.h>

namespace kerma
{
CudaDim::CudaDim()
: CudaDim(0, 0, 0)
{}

CudaDim::CudaDim(unsigned int x, unsigned int y, unsigned int z)
: x(x), y(y), z(z)
{}

CudaDim::CudaDim(const CudaDim &other)
: x(other.x), y(other.y), z(other.z)
{}

void
CudaDim::operator=(const CudaDim &other) 
{
  x = other.x;
  y = other.y;
  z = other.z;
}

bool
CudaDim::operator==(const CudaDim &other)
{
  return x == other.x && y == other.y && z == other.z;
}

/// TODO @todo Implement me
CudaDimError
validateGrid(CudaDim &dim)
{
  NOT_IMPLEMENTED_YET;
  return CudaDimError::Failure;
}

CudaDimError
vallidateGrid(CudaCompute &compute, CudaDim &dim)
{
  if ( compute == CudaCompute::Unknown )
    return CudaDimError::UnknownCompute;
  
  if ( compute < CudaCompute::cc_20 ) {
    if ( dim.x > 0xFFFF )
      return CudaDimError::InvalidDimX;
    if ( dim.y > 0xFFFF )
      return CudaDimError::InvalidDimY;
    if ( dim.z > 0x01 )
      return CudaDimError::UnsupportedDimZ;
  } else {
    if ( compute < CudaCompute::cc_30 && dim.x > 0xFFFF ||
         compute >= CudaCompute::cc_30 && dim.x > 0xFFFFFFFF )
      return CudaDimError::InvalidDimX;

    if ( dim.y > 0xFFFF )
      return CudaDimError::InvalidDimY;
    if ( dim.z > 0xFFFF )
      return CudaDimError::InvalidDimZ;
  }

  return CudaDimError::Success;
}

/// TODO @todo Implement me
CudaDimError
validateBlock(CudaDim &dim)
{
  NOT_IMPLEMENTED_YET;
  return CudaDimError::Failure;
}

CudaDimError
validateBlock(CudaCompute &compute, CudaDim &dim)
{
  if ( compute == CudaCompute::Unknown )
    return CudaDimError::UnknownCompute;
  
  if ( compute < CudaCompute::cc_20) {
    if ( dim.x > 512 )
      return CudaDimError::InvalidDimX;
    if ( dim.y > 512 )
      return CudaDimError::InvalidDimY;
  } else {
    if ( dim.x > 1024 )
      return CudaDimError::InvalidDimX;
    if ( dim.y > 1024 )
      return CudaDimError::InvalidDimY;
  }

  if ( dim.z > 64 )
    return CudaDimError::InvalidDimZ;

  return CudaDimError::Success;
}

} /* NAMESPACE kerma */
