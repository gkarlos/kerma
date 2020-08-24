#ifndef KERMA_PASS_MATERIALIZE_DIMS_H
#define KERMA_PASS_MATERIALIZE_DIMS_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "kerma/Base/Dim.h"

#include <memory>

namespace kerma {

/// This pass will go through the kernels and for each
/// kernel it will replace the gridDim.{x,y,z} and
/// blockDim.{x,y,z} values it encounters with concrete
/// values. These values as given as command line args
/// when the pass is run in Opt or passed as arguments
/// in the constructor. 
/// This is a transformation pass meaning the it _may_
/// modify the IR as a result.
class MaterializeDimsPass : public llvm::FunctionPass {

public:
  static char ID;
  // When the pass is run in Opt this is the 
  // only constructor that is used
  MaterializeDimsPass();

  MaterializeDimsPass(const Dim& Grid, const Dim& Block);
  MaterializeDimsPass(const Dim& Grid, const Dim& Block, llvm::Function &Kernel);
  MaterializeDimsPass(const Dim& Grid, const Dim& Block, const char *KernelName);

public:
  bool doInitialization(llvm::Module &M) override;
  bool doFinalization(llvm::Module &M) override;
  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool isTargeted();

private:
  bool analyzeKernel(llvm::Function &F) const;
  bool isKernel(llvm::Function &F);
  bool hasWork() const;
  bool hasTargetKernel() const;

private:
  std::vector<llvm::Function*> Kernels;
  llvm::Function *TargetKernelFun;
  const char *TargetKernelName;
  Dim Grid;
  Dim Block;
};

std::unique_ptr<MaterializeDimsPass>
createMaterializeDimsPass();

std::unique_ptr<MaterializeDimsPass>
createMaterializeDimsPass(const Dim& Grid, const Dim& Block);

std::unique_ptr<MaterializeDimsPass>
createMaterializeDimsPass(const Dim& Grid, const Dim& Block, llvm::Function &F);

} // namespace kerma

#endif // KERMA_PASS_MATERIALIZE_DIMS_H