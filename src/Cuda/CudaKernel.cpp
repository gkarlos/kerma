#include <kerma/Cuda/CudaKernel.h>

#include "llvm/IR/Function.h"

namespace kerma
{

CudaKernel::CudaKernel(llvm::Function &fn) 
: CudaKernel(fn, CudaSide::UKNOWN)
{}

CudaKernel::CudaKernel(llvm::Function &fn, CudaSide IRModuleSide)
: fn_(fn), 
  IRModuleSide_(IRModuleSide),
  numArgs_(0)
{}

void
CudaKernel::setIRModuleSide(CudaSide IRModuleSide)
{
  IRModuleSide_ = IRModuleSide;
}

CudaSide
CudaKernel::getIRModuleSide()
{
  return IRModuleSide_;
}


int
CudaKernel::getNumArgs()
{
  return numArgs_;
}


} /* NAMESPACE kerma */