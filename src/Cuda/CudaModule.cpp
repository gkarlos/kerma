//
// Created by gkarlos on 1/3/20.
//

#include "kerma/Cuda/Cuda.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Casting.h"
#include <kerma/Cuda/CudaModule.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <exception>

namespace kerma {



CudaModule::CudaModule( llvm::Module &hostModule, llvm::Module &deviceModule)
: hostModule_(hostModule), 
  deviceModule_(deviceModule)
{
  if ( deviceModule.getTargetTriple().empty())
    throw std::runtime_error("Internal Error: Target triple missing");

  try {
    std::string triple = deviceModule.getTargetTriple();
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

  llvm::NamedMDNode *MDcu = deviceModule_.getNamedMetadata("llvm.dbg.cu");
  if ( MDcu) {
    for ( const llvm::MDNode *node : MDcu->operands()) {
      if ( auto *DIcu = llvm::dyn_cast<llvm::DICompileUnit>(node)) {
        sourceFilename_ = DIcu->getFile()->getFilename();
        sourceDirectory_ = DIcu->getDirectory();
        if ( !sourceFilename_.empty() && !sourceDirectory_.empty())
          sourceFilenameFull_ = sourceDirectory_ + "/" + sourceFilename_;
      }
    }
  }
}

llvm::Module &
CudaModule::getHostModule() {
  return hostModule_;
}

llvm::Module &
CudaModule::getDeviceModule() {
  return deviceModule_;
}

CudaArch 
CudaModule::getArch() {
  return arch_;
}

bool 
CudaModule::is64bit() {
  return is64bit_;
}

bool 
CudaModule::is32bit() {
  return is32bit_;
}

void 
CudaModule::addKernel(CudaKernel &kernel) {
  kernels_.insert(kernel);
}

unsigned int
CudaModule::getNumberOfKernels()
{
  return kernels_.size();
}

std::set<CudaKernel> &
CudaModule::getKernels() {
  return kernels_;
}

const std::string &
CudaModule::getSourceFilename()
{
  return sourceFilename_;
}

const std::string &
CudaModule::getSourceDirectory()
{
  return sourceDirectory_;
}

const std::string &
CudaModule::getSourceFilenameFull()
{
  return sourceFilenameFull_;
}

}