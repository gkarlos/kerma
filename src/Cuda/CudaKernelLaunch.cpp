#include <kerma/Cuda/CudaKernel.h>
#include "kerma/Support/SourceCode.h"

namespace kerma
{

CudaKernelLaunch::CudaKernelLaunch(CudaKernel &kernel, int line)
: kernel_(kernel),
  line_(line)
{}

void
CudaKernelLaunch::setLaunchConfiguration(CudaKernelLaunchConfiguration &config)
{
  launchConfiguration_ = config;
}

CudaKernelLaunchConfiguration &
CudaKernelLaunch::getLaunchConfigutation()
{
  return launchConfiguration_;
}



} /* NAMESPACE kerma */