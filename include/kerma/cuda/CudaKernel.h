//
// Created by gkarlos on 1/5/20.
//

#ifndef KERMA_STATIC_ANALYSIS_CUDAKERNEL_H
#define KERMA_STATIC_ANALYSIS_CUDAKERNEL_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <kerma/cuda/CudaSupport.h>

namespace kerma
{
namespace cuda
{

class CudaKernel {

public:

  CudaKernel(llvm::Function* fn, kerma::cuda::CudaSide irModuleSide)
      : fn_(fn), irModuleSide_(irModuleSide)
  {}

  ~CudaKernel()=default;

  /*
   * Get a pointer to the llvm::Function for this kernel
   */
  llvm::Function* getFn()
  {
    return this->fn_;
  }

  /*
   * Get the side of the module where the kernel was detected (host, device)
   * Currently we only detect kernels from device IR.
   */
  kerma::cuda::CudaSide
  getModuleSide()
  {
    return this->irModuleSide_;
  }

  /*
   * Pretty-print info for this kernel
   */
  void pp(llvm::raw_ostream& s);

  inline bool operator==(const CudaKernel& other) const {
    return this->fn_ == other.fn_;
  }

  inline bool operator<(const CudaKernel& other) const {
    return this->fn_ < other.fn_;
  }

  inline bool operator>(const CudaKernel& other) const {
    return this->fn_ > other.fn_;
  }

private:
  llvm::Function* fn_;
  kerma::cuda::CudaSide irModuleSide_;
};

}
}

#endif // KERMA_STATIC_ANALYSIS_CUDAKERNEL_H
