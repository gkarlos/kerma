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
  signatureLineStart_(0),
  signatureLineEnd_(0),
  bodyLineStart_(0),
  bodyLineEnd_(0)
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
CudaKernel::setSignatureLineStart(unsigned int line)
{
  signatureLineStart_ = line;
}

void 
CudaKernel::setSignatureLineEnd(unsigned int line)
{
  signatureLineEnd_ = line;
}

void
CudaKernel::setSignatureLines(unsigned int start, unsigned int end)
{
  signatureLineStart_ = start;
  signatureLineEnd_ = end;
}

unsigned int 
CudaKernel::getSignatureLineStart()
{
  return signatureLineStart_;
}

unsigned int 
CudaKernel::getSignatureLineEnd()
{
  return signatureLineEnd_;
}

unsigned int
CudaKernel::getSignatureNumLines()
{
  if ( signatureLineEnd_ <= signatureLineStart_)
    return 0;
  return signatureLineEnd_ - signatureLineStart_;
}

void
CudaKernel::setBodyLineStart(unsigned int line)
{
  bodyLineStart_ = line;
}

void
CudaKernel::setBodyLineEnd(unsigned int line)
{
  bodyLineEnd_ = line;
}

void
CudaKernel::setBodyLines(unsigned int start, unsigned int end)
{
  bodyLineStart_ = start;
  bodyLineEnd_ = end;
}

unsigned int
CudaKernel::getBodyLineStart()
{
  return bodyLineStart_;
}

unsigned int
CudaKernel::getBodyLineEnd()
{
  return bodyLineEnd_;
}

unsigned int
CudaKernel::getBodyNumLines()
{
  if ( bodyLineEnd_ <= bodyLineStart_)
    return 0;
  return bodyLineEnd_ - bodyLineStart_;
}

unsigned int 
CudaKernel::getLineStart()
{
  return signatureLineStart_;
}

unsigned int 
CudaKernel::getLineEnd()
{
  return bodyLineEnd_;
}

unsigned int 
CudaKernel::getNumLines()
{
  if ( bodyLineEnd_ <= signatureLineStart_ )
    return 0;
  return bodyLineEnd_ - signatureLineStart_;
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