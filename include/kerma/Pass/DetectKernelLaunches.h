#ifndef KERMA_PASS_DETECT_KERNEL_LAUNCHES_H
#define KERMA_PASS_DETECT_KERNEL_LAUNCHES_H

#include <kerma/Config/Config.h>
#include <kerma/Pass/GenericOpts.h>
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Cuda/CudaModule.h>
#include <vector>

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/raw_ostream.h"

namespace kerma
{

#ifdef KERMA_TESTS_ENABLED
  #ifndef DETECT_KERNEL_LAUNCHES_PASS_UNIT_TEST_FRIENDS
    #define DETECT_KERNEL_LAUNCHES_PASS_UNIT_TEST_FRIENDS
  #endif
#endif

///
/// This Pass identifies Cuda kernel launch sites and 
/// launch configurations
///
/// Depends on:
///  * DetectKernelsPass
///  
class DetectKernelLaunchesPass : public llvm::ModulePass
{
public:
  static char ID;
  DetectKernelLaunchesPass();
  explicit DetectKernelLaunchesPass(kerma::CudaModule &cudaModule);

public:
  bool runOnModule(llvm::Module &M) override;
  bool doInitialization(llvm::Module &M) override;
  bool doFinalization(llvm::Module &M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  
public:
  /*!
   * @brief Retrieve all the launches in the module
   */
  std::vector<kerma::CudaKernelLaunch*> getLaunches();
  
  /*!
   * @brief Retrieve the launches associated with a specific kernel 
   * @param kernel [in] A CudaKernel
   */
  std::vector<kerma::CudaKernelLaunch*> getLaunchesForKernel(kerma::CudaKernel &kernel);

private:
  /// Map kernels with their launches
  std::map<CudaKernel, std::vector<CudaKernelLaunch>> map_;

#ifdef KERMA_TESTS_ENABLED
  DETECT_KERNEL_LAUNCHES_PASS_UNIT_TEST_FRIENDS
#endif
};

} /* NAMESPACE kerma */

#endif /* KERMA_PASS_DETECT_KERNEL_LAUNCHES_H */