#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/NVVM.h>
#include <string>

namespace kerma
{


AddressSpace::AddressSpace(const std::string& name, int code)
: name_(name), 
  code_(code)
{}

AddressSpace AddressSpace::CODE     ("code",     0);
AddressSpace AddressSpace::GENERIC  ("generic",  0);
AddressSpace AddressSpace::GLOBAL   ("global",   1);
AddressSpace AddressSpace::SHARED   ("shared",   3);
AddressSpace AddressSpace::CONSTANT ("constant", 4);
AddressSpace AddressSpace::LOCAL    ("local",    5);
AddressSpace AddressSpace::UNKNOWN  ("unknown", -1);


bool AddressSpace::operator==(AddressSpace &other) {
  return name_.compare(other.name_) == 0 && code_ == other.code_;
}

bool AddressSpace::operator!=(AddressSpace &other) {
  return !operator==(other);
}

const std::string& AddressSpace::getName() {
  return name_;
}

const int AddressSpace::getCode() {
  return code_;
}


bool isDeviceModule(llvm::Module& module)
{
  return module.getTargetTriple().find("nvptx") != std::string::npos;
}

bool isHostModule(llvm::Module& module)
{
  return !isDeviceModule(module);
}

CudaSide getIRModuleSide(llvm::Module &module)
{
  return isDeviceModule(module)? CudaSide::DEVICE : 
        (isHostModule(module)? CudaSide::HOST : CudaSide::Unknown);
}

}