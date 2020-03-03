#ifndef KERMA_PASS_DETECT_KERNELS_H
#define KERMA_PASS_DETECT_KERNELS_H

#include "kerma/Cuda/CudaKernel.h"
#include "kerma/Cuda/CudaProgram.h"
#include "kerma/Support/PrettyPrintable.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/raw_ostream.h"

namespace kerma {

/*
 * An LLVM Pass that detects the kernel functions in a CudaProgram 
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
  explicit DetectKernelsPass(kerma::CudaProgram &program);

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
   * @brief Check if a CudaProgram is attached to this pass
   */
  bool hasCudaProgramAttached();

  /*
   * @brief Retrieve the CudaProgram associated with the pass.
   *        Return nullptr when no program is attached
   */
  CudaProgram *getCudaProgram();

  /*
   * @brief Attach a CudaProgram to this Pass
   * 
   * Once a CudaProgram is attached, subsequent calls are no-ops
   *
   * Attaching a CudaProgram will populate the program with the Kernels found
   * by the pass (if any). It is up to the user to make sure that the CudaProgram 
   * passed corresponds to the same IR the pass was run on. 
   * If in doubt, runOnModule() can be called after attaching the CudaProgram.
   *
   * The above is mostly relevant when the pass is run with opt.
   * When run programmatically (through a PassManager) we can attach a CudaProgram
   * directly in the constructor:
   * <code>
   *  CudaProgram program(...);
   *  legacy::PassManager PM;
   *  DetectKernelsPass *detectKernels = new DetectKernelsPass(&program);
   *  PM.add(detectKernels);
   *  PM.run( program.getDeviceModule() )
   * </code>
   *
   * @param program - A CudaProgram
   * @return true  - program attached
   *         false - program not attached 
   */
  bool attachCudaProgram(CudaProgram &program);

private:
  CudaProgram *program_;
  std::set<CudaKernel> kernels_;
};

} /* NAMESPACE kerma */

#endif