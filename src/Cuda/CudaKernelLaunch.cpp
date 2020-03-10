#include <kerma/Cuda/CudaKernel.h>
#include "kerma/Support/SourceCode.h"
#include "llvm/IR/Instructions.h"

namespace kerma
{

CudaKernelLaunch::CudaKernelLaunch(CudaKernel &kernel, int line)
: kernel_(kernel),
  cudaLaunchKernelCall_(nullptr),
  line_(line)
{}

CudaKernelLaunch::~CudaKernelLaunch() = default;

CudaKernel &
CudaKernelLaunch::getKernel()
{
  return kernel_;
}

CudaKernelLaunchConfiguration &
CudaKernelLaunch::getLaunchConfigutation()
{
  return launchConfiguration_;
}

void
CudaKernelLaunch::setLaunchConfiguration(CudaKernelLaunchConfiguration &config)
{
  launchConfiguration_ = config;
}

llvm::CallInst *
CudaKernelLaunch::setCudaLaunchKernelCall(llvm::CallInst *kernelCall)
{
  llvm::CallInst *res = cudaLaunchKernelCall_;
  cudaLaunchKernelCall_ = kernelCall;
  return res;
}

llvm::CallInst *
CudaKernelLaunch::getCudaLaunchKernelCall()
{
  return cudaLaunchKernelCall_;
}

unsigned int
CudaKernelLaunch::getLine()
{
  return line_;
}

unsigned int
CudaKernelLaunch::setLine(unsigned int line)
{
  unsigned int res = line_;
  line_ = line;
  return res;
}

OPTIONAL<bool>
CudaKernelLaunch::inLoop()
{
  return inLoop_;
}

OPTIONAL<bool>
CudaKernelLaunch::setInLoop(bool inLoop)
{
  OPTIONAL<bool> res(inLoop_);
  inLoop_ = inLoop;
  return res;
}

OPTIONAL<bool>
CudaKernelLaunch::unsetInLoop()
{
  OPTIONAL<bool> noVal;
  OPTIONAL<bool> res(inLoop_);
  inLoop_ = {};
  return res;
}


OPTIONAL<bool>
CudaKernelLaunch::inThenBranch()
{
  return inThen_;
}

OPTIONAL<bool>
CudaKernelLaunch::inElseBranch()
{
  return inElse_;
}

static OPTIONAL<bool>
__optionalOR(OPTIONAL<bool> _l, OPTIONAL<bool> _r)
{
  OPTIONAL<bool> res;
  if ( _l && _r )
    res = _l.value() || _r.value();
  return res;
}

OPTIONAL<bool>
CudaKernelLaunch::inBranch()
{
  return __optionalOR(inThen_, inElse_);
}

OPTIONAL<bool>
CudaKernelLaunch::setInThenBranch(bool inThenBranch)
{
  OPTIONAL<bool> res(inThen_);
  inThen_ = inThenBranch;
  return res;
}

OPTIONAL<bool>
CudaKernelLaunch::setInElseBranch(bool inElseBranch)
{
  OPTIONAL<bool> res(inElse_);
  inElse_ = inElseBranch;
  return res;
}

OPTIONAL<bool>
CudaKernelLaunch::unsetInBranch()
{
  OPTIONAL<bool> res = __optionalOR(inThen_, inElse_);
  inThen_ = false;
  inElse_ = false;
  return res;
}

} /* NAMESPACE kerma */