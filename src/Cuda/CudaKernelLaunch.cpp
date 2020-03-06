#include "kerma/Support/SourceCode.h"
#include <kerma/Cuda/CudaKernel.h>


namespace kerma
{

CudaKernelLaunch::CudaKernelLaunch(CudaKernel *kernel, int line)
: CudaKernelLaunch(kernel, nullptr, line)
{}

CudaKernelLaunch::CudaKernelLaunch(CudaKernel *kernel, CudaKernelLaunchConfiguration *config, int line)
: kernel_(kernel),
  launchConfiguration_(config),
  line_(line)
{}

CudaKernelLaunch::~CudaKernelLaunch() = default;

void
CudaKernelLaunch::setLaunchConfiguration(CudaKernelLaunchConfiguration *config)
{
  launchConfiguration_ = config;
}

CudaKernelLaunchConfiguration *
CudaKernelLaunch::getLaunchConfigutation()
{
  return launchConfiguration_;
}

} /* NAMESPACE kerma */