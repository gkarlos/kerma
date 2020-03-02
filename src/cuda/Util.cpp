
#include <kerma/cuda/CudaSupport.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Demangle/Demangle.h>

namespace kerma
{
namespace cuda
{


/*
 * @brief Get a string value of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia Architecture
 */


/*
 * @brief Get the name of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia architecture
 */


std::string cudaArchName(Compute cc)
{
  switch ( cc) {
  case Compute::CC30:
  case Compute::CC32:
  case Compute::CC35:
  case Compute::CC37:
    return "Kepler";
  case Compute::CC50:
  case Compute::CC52:
  case Compute::CC53:
    return "Maxwell";
  case Compute::CC60:
  case Compute::CC61:
  case Compute::CC62:
    return "Pascal";
  case Compute::CC70:
  case Compute::CC72:
  case Compute::CC75:
    return "Turing";
  default:
    return "Unknown";
  }
}



std::string cudaSideToString(CudaSide side)
{
  switch (side) {
  case CudaSide::HOST:
    return "host";
  case CudaSide::DEVICE:
    return "device";
  default:
    return "unknown";
  }
}

} /// NAMESPACE cuda

bool
isDeviceModule(llvm::Module &module)
{
  return module.getTargetTriple().find("nvptx") != std::string::npos;
}

bool
isHostModule(llvm::Module &module)
{
  return !isDeviceModule(module);
}

cuda::CudaSide
getIRModuleSide(llvm::Module &module)
{
  return isDeviceModule(module) ? cuda::CudaSide::DEVICE : cuda::CudaSide::HOST;
}


} /// NAMESPACE kerma