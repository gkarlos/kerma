#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Support/LLVMFunctionShorthands.h>
#include <kerma/Support/Demangle.h>
#include <kerma/Support/SourceCode.h>

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
  signatureLineStart_(SRC_LINE_UNKNOWN),
  signatureLineEnd_(SRC_LINE_UNKNOWN),
  bodyLineStart_(SRC_LINE_UNKNOWN),
  bodyLineEnd_(SRC_LINE_UNKNOWN)
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
  /// If the new start line is greater than old end line then
  /// the old end line cannot be valid anymore
  if ( signatureLineEnd_ != SRC_LINE_UNKNOWN && signatureLineStart_ > signatureLineEnd_)
    signatureLineEnd_ = SRC_LINE_UNKNOWN;
  
  if ( line > bodyLineStart_ || line > bodyLineEnd_ ) {
    bodyLineStart_ = SRC_LINE_UNKNOWN;
    bodyLineEnd_ = SRC_LINE_UNKNOWN;
  } 
}

void
CudaKernel::setSignatureLineEnd(unsigned int line)
{
  signatureLineEnd_ = line;

  /// An end line less than the start line cannot be valid
  if ( line < signatureLineStart_ )
    signatureLineEnd_ = SRC_LINE_UNKNOWN;
  
  /// If body lines where set and signate end inteleaves with them,
  /// then body lines get invalidated
  if ( line > bodyLineStart_ || line > bodyLineEnd_ ) {
    bodyLineStart_ = SRC_LINE_UNKNOWN;
    bodyLineEnd_ = SRC_LINE_UNKNOWN;
  } 
}

void
CudaKernel::setSignatureLines(unsigned int start, unsigned int end)
{
  setSignatureLineStart(start);
  setSignatureLineEnd(end);
}

int 
CudaKernel::getSignatureLineStart()
{
  return signatureLineStart_;
}

int 
CudaKernel::getSignatureLineEnd()
{
  return signatureLineEnd_;
}

int
CudaKernel::getSignatureNumLines()
{
  if ( signatureLineStart_ == SRC_LINE_UNKNOWN 
    || signatureLineEnd_ == SRC_LINE_UNKNOWN 
    || signatureLineEnd_ < signatureLineStart_ )
    return 0;
  return 1 + signatureLineEnd_ - signatureLineStart_;
}

void
CudaKernel::setBodyLineStart(unsigned int line)
{
  bodyLineStart_ = line;

  if ( signatureLineEnd_ != SRC_LINE_UNKNOWN && line < signatureLineEnd_)
    bodyLineStart_ = SRC_LINE_UNKNOWN;

  if ( bodyLineEnd_ != SRC_LINE_UNKNOWN && bodyLineStart_ > bodyLineEnd_)
    bodyLineEnd_ = SRC_LINE_UNKNOWN;
}

void
CudaKernel::setBodyLineEnd(unsigned int line)
{
  bodyLineEnd_ = line;
  
  /// An end line less than the start line cannot be valid
  if ( line < bodyLineStart_ )
    bodyLineEnd_ = SRC_LINE_UNKNOWN;
}

void
CudaKernel::setBodyLines(unsigned int start, unsigned int end)
{
  setBodyLineStart(start);
  setBodyLineEnd(end);
}

int
CudaKernel::getBodyLineStart()
{
  return bodyLineStart_;
}

int
CudaKernel::getBodyLineEnd()
{
  return bodyLineEnd_;
}

int
CudaKernel::getBodyNumLines()
{ 
  if ( bodyLineStart_ == SRC_LINE_UNKNOWN 
    || bodyLineEnd_ == SRC_LINE_UNKNOWN 
    || bodyLineEnd_ < bodyLineStart_)
    return 0;

  return 1 + bodyLineEnd_ - bodyLineStart_;
}

int 
CudaKernel::getLineStart()
{
  return signatureLineStart_;
}

int 
CudaKernel::getLineEnd()
{
  return bodyLineEnd_;
}

int 
CudaKernel::getNumLines()
{
  if ( signatureLineStart_ == SRC_LINE_UNKNOWN
    || bodyLineEnd_ == SRC_LINE_UNKNOWN
    || bodyLineEnd_ < signatureLineStart_ )
    return 0;

  return 1 + bodyLineEnd_ - signatureLineStart_;
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