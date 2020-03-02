#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Support/LLVMFunctionShorthands.h>
#include <kerma/Support/Demangle.h>

#include "llvm/IR/Argument.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"

namespace kerma
{

/// Constructors

CudaKernel::CudaKernel(llvm::Function &fn) 
: CudaKernel(fn, CudaSide::Unknown)
{}

CudaKernel::CudaKernel(llvm::Function &fn, CudaSide IRModuleSide)
: fn_(fn), 
  IRModuleSide_(IRModuleSide),
  numArgs_(0)
{}

/// Operators

bool
CudaKernel::operator==(const CudaKernel &other) const
{
  return static_cast<llvm::Value*>(&fn_) == static_cast<llvm::Value*>(&(other.fn_));
}

bool
CudaKernel::operator<(const CudaKernel &other) const
{
  return static_cast<llvm::Value*>(&fn_) < static_cast<llvm::Value*>(&(other.fn_));
}

bool
CudaKernel::operator>(const CudaKernel &other) const
{
  return static_cast<llvm::Value*>(&fn_) > static_cast<llvm::Value*>(&(other.fn_));
}

/// API

llvm::Function &
CudaKernel::getFn()
{
  return fn_;
}

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
  return getFnNumArgs(fn_);
}

void 
CudaKernel::pp(llvm::raw_ostream& os) {
  std::string demangled = demangleFn(&fn_);
  os << demangled
     << ((demangled != fn_.getName())? " (demangled)" : "") << "\n"
     << " " << u8"└" << " In " << getCudaSideToStr(IRModuleSide_)
     << "-side module:" << (fn_.getParent())->getName() << "\n";
}


} /* NAMESPACE kerma */