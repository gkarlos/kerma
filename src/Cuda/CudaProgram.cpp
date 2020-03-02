//
// Created by gkarlos on 1/3/20.
//

#include "kerma/Cuda/Cuda.h"
#include <kerma/Cuda/CudaProgram.h>

#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <exception>

namespace kerma {



CudaProgram::CudaProgram( llvm::Module &hostModule, llvm::Module &deviceModule)
: hostModule_(hostModule), deviceModule_(deviceModule)
{
  if ( deviceModule.getTargetTriple().empty())
    throw std::runtime_error("Internal Error: Target triple missing");

  try {
    std::string const& triple = deviceModule.getTargetTriple();
    std::string name = deviceModule.getName();

    name = name.substr(0, name.size() - 3);

    std::string part = triple.substr(0, triple.find("-"));
    this->is64bit_ = part == "nvptx64";
    this->is32bit_ = !this->is64bit_;

    std::size_t previous = 0, current = name.find("-");
    while (current != std::string::npos) {
      previous = current + 1;
      current = name.find("-", previous);
    }

    this->arch_ = getCudaArch( name.substr(previous, current - previous));

  } catch(...) {
    throw;
  }
}

llvm::Module &
CudaProgram::getHostModule() {
  return hostModule_;
}

llvm::Module &
CudaProgram::getDeviceModule() {
  return deviceModule_;
}

CudaArch 
CudaProgram::getArch() {
  return arch_;
}

bool 
CudaProgram::is64bit() {
  return is64bit_;
}

bool 
CudaProgram::is32bit() {
  return is32bit_;
}

}