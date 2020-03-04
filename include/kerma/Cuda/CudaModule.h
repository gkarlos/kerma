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

class CudaModule {
public:

  CudaModule ( llvm::Module &hostModule, llvm::Module &deviceModule);

  ~CudaModule()=default;

  /*
   * Retrieve the host-side LLVM Module for this program
   */
  llvm::Module &getHostModule();

  /*
   * Retrieve the device-side LLVM Module for this program
   */
  llvm::Module &getDeviceModule();

  /*
   * Get the Nvidia arch as defined in the LLVM IR
   */
  CudaArch getArch();

  /*
   * Check if the CudaModule (as defined by the host and device IR) targets 64-bit arch
   */
  bool is64bit();

  /*
   * Check if the CudaModule (as defined by the host and device IR) targets 32-bit arch
   */
  bool is32bit();

  /*
   * Attach a CudaKernel to this CudaModule
   */
  void addKernel(CudaKernel &kernel);

  /*
   * Retrieve the CudaKernel(s) attached to this CudaModule
   */
  std::set<CudaKernel> &getKernels();

  /*
   * Retrieve the number of kernels attached to this CudaModule
   */
  unsigned int getNumberOfKernels();

  /*
   * Retrieve the name of the source code file associated with this CudaModule
   * (as defined by the host and device IR)
   */
  std::string getSourceFilename();

private:
  llvm::Module &hostModule_;
  llvm::Module &deviceModule_;
  std::set<CudaKernel> kernels_;
  CudaArch arch_;
  bool is64bit_;
  bool is32bit_;
};

} // NAMESPACE kerma


#endif // KERMA_STATIC_ANALYSIS_CUDAPROGRAM_H
