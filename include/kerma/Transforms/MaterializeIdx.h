#ifndef KERMA_PASS_MATERIALIZE_IDX_H
#define KERMA_PASS_MATERIALIZE_IDX_H

#include "kerma/Base/Index.h"

#include <llvm/IR/Function.h>
#include <llvm/Pass.h>

#include <memory>

namespace kerma {

/// This pass will go through the kernels and for each
/// kernel it will replace the blockIdx.{x,y,z} and 
/// threadIdx.{x,y,z} values it encounters with concrete
/// vallues. These values are given as comment line args
/// when the pass ir run in Opt, or passes as arguments
/// in the constructor.
/// This is a transformation pass meaning that it _may_
/// the IR as a result.
/// This pass should have no effect on non-Cuda modules!
class MaterializeIdxPass : public llvm::FunctionPass {

public:
  static char ID;
  // When the pass is run in Opt this is the 
  // only constructor that is used 
  MaterializeIdxPass();

  MaterializeIdxPass(const Index& Block, const Index& Thread);
  MaterializeIdxPass(const Index& Block, const Index& Thread, llvm::Function &Kernel);
  MaterializeIdxPass(const Index& Block, const Index& Thread, const char *KernelName);

public:
  bool doInitialization(llvm::Module &M) override;
  bool doFinalization(llvm::Module &M) override;
  bool runOnFunction(llvm::Function &F) override;
  // void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

public:
  bool hasTargetKernel() const;

private:
  bool analyzeKernel(llvm::Function &F) const;
  bool isKernel(llvm::Function &F);
  bool hasWork() const;

private:
  std::vector<llvm::Function*> Kernels;
  llvm::Function *TargetKernelFun;
  const char *TargetKernelName;
  Index Block;
  Index Thread;
};

std::unique_ptr<MaterializeIdxPass> createMaterializeIdxPass();

std::unique_ptr<MaterializeIdxPass>
createMaterializeIdxPass(const Index& Block, const Index& Thread);

std::unique_ptr<MaterializeIdxPass>
createMaterializeIdxPass(const Index& Block, const Index& Thread, llvm::Function& TargetKernel);

std::unique_ptr<MaterializeIdxPass>
createMaterializeIdxPass(const Index& Block, const Index& Thread, const char *TargetKernelName);


} // namespace kerma

#endif // // KERMA_PASS_MATERIALIZE_DIMS_H