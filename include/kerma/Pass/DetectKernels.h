#ifndef KERMA_PASS_DETECT_KERNELS_H
#define KERMA_PASS_DETECT_KERNELS_H

#include "kerma/Cuda/CudaKernel.h"
#include "kerma/Cuda/CudaModule.h"
#include "kerma/Support/PrettyPrintable.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/raw_ostream.h"

namespace kerma {

/*
 * An LLVM Pass that detects the kernel functions in a CudaModule 
 * After running the pass users can retrieve a set of CudaKernel(s)
 * or query to check if an llvm::Function is a CUDA kernel definition.
 *
 * This Pass is meant to be run early in the Kerma pipeline.
 */
class DetectKernelsPass : public llvm::ModulePass
{

public:
  static char ID;
  DetectKernelsPass();
  explicit DetectKernelsPass(kerma::CudaModule &cudaModule);

public:
  bool runOnModule(llvm::Module &M) override;
  bool doInitialization(llvm::Module &M) override;
  bool doFinalization(llvm::Module& M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;

public:
  /*
   * @brief Retrieve the CUDA kernels found after running the Pass
   */
  std::set<CudaKernel> getKernels();

  /*
   * @brief Write the CUDA kernels found to a provided set
   *
   * @param [in] kernels An std::set to write the kernels found
   */
  void getKernels(std::set<CudaKernel> &kernels);

  /*
   * @brief Check if an llvm::Function is a CUDA kernel
   */
  bool isKernel(llvm::Function &F);

  /*
   * @brief Check if a CudaModule is attached to this pass
   */
  bool hasCudaModuleAttached();

  /*
   * @brief Retrieve the CudaModule associated with the pass.
   *        Return nullptr when no program is attached
   */
  CudaModule *getCudaModule();

  /*
   * @brief Attach a CudaModule to this Pass
   * 
   * Once a CudaModule is attached, subsequent calls are no-ops
   *
   * Attaching a CudaModule will populate the program with the Kernels found
   * by the pass (if any). It is up to the user to make sure that the CudaModule 
   * passed corresponds to the same IR the pass was run on. 
   * If in doubt, runOnModule() can be called after attaching the CudaModule.
   *
   * The above is mostly relevant when the pass is run with opt.
   * When run programmatically (through a PassManager) we can attach a CudaModule
   * directly in the constructor:
   * <code>
   *  CudaModule program(...);
   *  legacy::PassManager PM;
   *  DetectKernelsPass *detectKernels = new DetectKernelsPass(&program);
   *  PM.add(detectKernels);
   *  PM.run( program.getDeviceModule() )
   * </code>
   *
   * @param cudaModule - A CudaModule
   * @return true  - program attached
   *         false - program not attached 
   */
  bool attachCudaModule(CudaModule &cudaModule);

private:
  CudaModule *cudaModule_;
  std::set<CudaKernel> kernels_;
};

} /* NAMESPACE kerma */

#endif