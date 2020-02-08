
#include <kerma/cuda/CudaSupport.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Demangle/Demangle.h>

namespace kerma
{
namespace cuda
{
/*
 * @brief Get a string value for a Cuda Compute Capability
 *
 * @param [in] c A Cuda Compute Capability
 * @param [in] full Get a string with a full description
 */
std::string cudaComputeToString(Compute c, bool full)
{
  switch (c) {
  case Compute::CC30:
    return full? "Compute Capability 3.0" : "CC30";
  case Compute::CC32:
    return full? "Compute Capability 3.2" : "CC32";
  case Compute::CC35:
    return full? "Compute Capability 3.5" : "CC35";
  case Compute::CC37:
    return full? "Compute Capability 3.7" : "CC37";
  case Compute::CC50:
    return full? "Compute Capability 5.0" : "CC50";
  case Compute::CC52:
    return full? "Compute Capability 5.2" : "CC52";
  case Compute ::CC53:
    return full? "Compute Capability 5.3" : "CC53";
  case Compute::CC60:
    return full? "Compute Capability 6.0" : "CC60";
  case Compute::CC61:
    return full? "Compute Capability 6.1" : "CC61";
  case Compute::CC62:
    return full? "Compute Capability 6.2" : "CC62";
  case Compute::CC70:
    return full? "Compute Capability 7.0" : "CC70";
  case Compute::CC72:
    return full? "Compute Capability 7.2" : "CC72";
  case Compute::CC75:
    return full? "Compute Capability 7.5" : "CC75";
  default:
    return full? "Unknown Compute Capability" : "Unknown";
  }
}

/*
 * @brief Get a string value of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia Architecture
 */
std::string cudaArchToString(Arch arch)
{
  switch ( arch) {
  case Arch::SM30:
    return "sm_30";
  case Arch::SM32:
    return "sm_32";
  case Arch::SM35:
    return "sm_35";
  case Arch::SM50:
    return "sm_50";
  case Arch::SM52:
    return "sm_52";
  case Arch::SM53:
    return "sm_53";
  case Arch::SM60:
    return "sm_60";
  case Arch::SM61:
    return "sm_61";
  case Arch::SM62:
    return "sm_62";
  case Arch::SM70:
    return "sm_70";
  case Arch::SM72:
    return "sm_72";
  case Arch::SM75:
    return "sm_75";
  default:
    return "Unknown Architecture";
  }
}

/*
 * @brief Get the name of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia architecture
 */
std::string cudaArchName(Arch arch)
{
  switch ( arch) {
  case Arch::SM30:
  case Arch::SM32:
  case Arch::SM35:
    return "Kepler";
  case Arch::SM50:
  case Arch::SM52:
  case Arch::SM53:
    return "Maxwell";
  case Arch::SM60:
  case Arch::SM61:
  case Arch::SM62:
    return "Pascal";
  case Arch::SM70:
  case Arch::SM72:
    return "Volta";
  case Arch::SM75:
    return "Turing";
  default:
    return "Unknown";
  }
}

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

Arch archFromString(const std::string& arch) {
  if ( arch == "sm_30")
    return Arch::SM30;
  else if ( arch == "sm_32")
    return Arch::SM32;
  else if ( arch == "sm_35")
    return Arch::SM35;
  else if ( arch == "sm_50")
    return Arch::SM50;
  else if ( arch == "sm_52")
    return Arch::SM52;
  else if ( arch == "sm_53")
    return Arch::SM53;
  else if ( arch == "sm_60")
    return Arch::SM60;
  else if ( arch == "sm_61")
    return Arch::SM61;
  else if ( arch == "sm_62")
    return Arch::SM62;
  else if ( arch == "sm_70")
    return Arch::SM70;
  else if ( arch == "sm_72")
    return Arch::SM72;
  else if ( arch == "sm_75")
    return Arch::SM75;
  else
    return Arch::Unknown;
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

// #if LLVM_VERSION_MAJOR < 9
 
//   int status;
//   char *p = abi::__cxa_demangle( this->fn_->getName().str().c_str(), nullptr, nullptr, &status);
//   std::string demangled(p);
//   free(p);

// #else
  
//   std::string demangled = llvm::demangle(this->fn_->getName());

// #endif

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