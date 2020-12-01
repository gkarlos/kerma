#ifndef KERMA_TRANSFORMS_META_KERNELS_H
#define KERMA_TRANSFORMS_META_KERNELS_H

#include "kerma/Base/Mode.h"
#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

namespace kerma {

/// This pass is meant to be run on the MetaDriver module
class MetaKernelFullPass : public llvm::ModulePass {
private:
  KernelInfo &KI;
  AssumptionInfo &AI;
  enum Mode Mode;
  Index TargetBlock;
  Index TargetThread;
  unsigned TargetWarp;

public:
  static char ID;
  MetaKernelFullPass(KernelInfo &KI, AssumptionInfo &AI, enum Mode=BLOCK,
                     const Index &TargetBlock = Index(0, 0, 0),
                     const Index &TargetThread = Index(0, 0, 0),
                     unsigned TargetWarp = 0);
  bool runOnModule(llvm::Module &M) override;
  llvm::StringRef getPassName() const override { return "MetaKernelFullPass"; }
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace kerma

#endif