#ifndef KERMA_TRANSFORMS_PREPARE_META_DRIVER_H
#define KERMA_TRANSFORMS_PREPARE_META_DRIVER_H

#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

namespace kerma {

class PrepareMetaDriverPass : public llvm::ModulePass {
public:
  static char ID;
  PrepareMetaDriverPass();
  bool runOnModule(llvm::Module &M) override;
  llvm::StringRef getPassName() const override { return "PrepareMetaDriverPass"; }
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_PREPARE_META_DRIVER_H