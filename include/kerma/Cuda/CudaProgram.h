//
// Created by gkarlos on 1/3/20.
//

#ifndef KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H
#define KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H

#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Cuda/Cuda.h>
#include <llvm/IR/Module.h>
#include <set>

namespace kerma {


// struct CudaKernelComparator {
//   // or maybe, if this logic already exists:
//   bool operator()( CudaKernel* lhs, CudaKernel* rhs) const {
//     return lhs->getFn() < rhs->getFn();
//   }
// };

class CudaProgram {
public:

  CudaProgram ( llvm::Module &hostModule, llvm::Module &deviceModule);

  ~CudaProgram()=default;

  /*
   * Retrieve the host-side Module for this program
   */
  llvm::Module &getHostModule();

  /*
   * Retrieve the device-side Module for this program
   */
  llvm::Module &getDeviceModule();

  /*
   * Get the Nvidia arch as defined in the LLVM IR
   */
  CudaArch getArch();


  bool is64bit();
  bool is32bit();


  void addKernel(CudaKernel &kernel) {
    this->kernels_.insert(kernel);
  }

  std::set<CudaKernel> &getKernels() {
    return this->kernels_;
  }

private:
  llvm::Module &hostModule_;
  llvm::Module &deviceModule_;
  // Use a custom comparator to insert in order to avoid duplicates, since the
  // pass (DetectKernels) creates a new CudaKernel object for each kernel fn it
  // detects and the pass could potentially run multiple times
  std::set<CudaKernel> kernels_;
  CudaArch arch_;
  bool is64bit_;
  bool is32bit_;
};

} // NAMESPACE kerma


#endif // KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H
