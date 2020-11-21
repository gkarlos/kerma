#ifndef KERMA_TRANSFORMS_STRIP_ANNOTATIONS_H
#define KERMA_TRANSFORMS_STRIP_ANNOTATIONS_H

#include "kerma/Analysis/DetectKernels.h"
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

namespace kerma {

class StripAnnotationsPass : public llvm::ModulePass {
public:
  static char ID;
  StripAnnotationsPass(KernelInfo &KI);
  bool runOnModule(llvm::Module &M) override;
  llvm::StringRef getPassName() const override { return "StripAnnotationsPass"; }
private:
  KernelInfo &KI;
};

} // namespace kerma

#endif