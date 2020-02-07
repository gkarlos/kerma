//
// Created by gkarlos on 1/3/20.
//

#ifndef KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H
#define KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H

#include <kerma/cuda/CudaSupport.h>
#include <kerma/cuda/CudaKernel.h>
#include <llvm/IR/Module.h>
#include <set>

namespace kerma { namespace cuda {


struct CudaKernelComparator {
  // or maybe, if this logic already exists:
  bool operator()( CudaKernel* lhs, CudaKernel* rhs) const {
    return lhs->getFn() < rhs->getFn();
  }
};

class CudaProgram {
public:
  CudaProgram() : CudaProgram(nullptr, nullptr, Arch::Unknown)
  {}

  CudaProgram ( llvm::Module* hostModule, llvm::Module* deviceModule);

  CudaProgram( llvm::Module* hostModule, llvm::Module* deviceModule, Arch arch)
  : CudaProgram(hostModule, deviceModule, arch, false)
  {}

  CudaProgram( llvm::Module *hostModule, llvm::Module *deviceModule, Arch arch, bool is64bit)
  : hostModule_(hostModule), deviceModule_(deviceModule), arch_(arch),
    is64bit_(is64bit), is32bit_(!is64bit)
  {}

  ~CudaProgram()=default;

  llvm::Module* getHostModule() {
    return this->hostModule_;
  }

  llvm::Module* getDeviceModule() {
    return this->deviceModule_;
  }

  cuda::Arch getArch() {
    return this->arch_;
  }

  bool is64bit() {
    return this->is64bit_;
  }

  bool is32bit() {
    return this->is32bit_;
  }

  void addKernel(CudaKernel* kernel) {
    this->kernels_.insert(kernel);
  }

  std::set<CudaKernel*, CudaKernelComparator>& getKernels() {
    return this->kernels_;
  }

private:
  llvm::Module* hostModule_;
  llvm::Module* deviceModule_;
  // Use a custom comparator to insert in order to avoid duplicates, since the
  // pass (DetectKernels) creates a new CudaKernel object for each kernel fn it
  // detects and the pass could potentially run multiple times
  std::set<CudaKernel*, CudaKernelComparator> kernels_;
  cuda::Arch arch_;
  bool is64bit_;
  bool is32bit_;
};

} // NAMESPACE cuda
} // NAMESPACE kerma


#endif // KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H
