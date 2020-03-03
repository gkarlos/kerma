#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Support/LLVMFunctionShorthands.h>
#include <kerma/Support/Demangle.h>
#include <ostream>

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
  mangledName_(fn.getName()),
  name_(demangleFnWithoutArgs(fn)),
  lineStart_(0),
  lineEnd_(0)
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
  return fn_.arg_size();
}

std::string&
CudaKernel::getName() {
  return name_;
}

std::string&
CudaKernel::getMangledName()
{
  return mangledName_;
}

void 
CudaKernel::setLineStart(unsigned int line)
{
  lineStart_ = line;
}

void 
CudaKernel::setLineEnd(unsigned int line)
{
  lineEnd_ = line;
}

int 
CudaKernel::getLineStart()
{
  return lineStart_;
}

int 
CudaKernel::getLineEnd()
{
  return lineEnd_;
}

int CudaKernel::getNumLines()
{
  if ( lineEnd_ <= lineStart_ )
    return 0;
  
  return lineEnd_ - lineStart_;
}

void 
CudaKernel::pp(llvm::raw_ostream& os) {
  std::string demangled = demangleFn(fn_);
  os << demangled
     << " " << u8"└" << " In " << getCudaSideStr(IRModuleSide_) << "\n";
    //  << "-side module:" << (fn_.getParent())->getName() << "\n";
}

void 
CudaKernel::pp(std::ostream& os) {
  std::string demangled = demangleFn(fn_);

  os << demangled
     << " " << u8"└" << " In " << getCudaSideStr(IRModuleSide_) << "\n";
    //  << "-side module:" << (fn_.getParent())->getName() << "\n";
}


} /* NAMESPACE kerma */