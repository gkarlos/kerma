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

void
CudaDim::operator=(CudaDim &other) 
{
  x = other.x;
  y = other.y;
  z = other.z;
}

bool
CudaDim::operator==(CudaDim &other)
{
  return x == other.x && y == other.y && z == other.z;
}

int
valiadateGrid(CudaDim &dim)
{
  NOT_IMPLEMENTED_YET;
  /// TODO Implement me
  return -1;
}

/// TODO @todo Implement me
int
validateBlock(CudaDim &dim)
{
  NOT_IMPLEMENTED_YET;
  return -1;
}

} /* NAMESPACE kerma */
