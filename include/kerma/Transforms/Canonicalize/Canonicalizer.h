#ifndef KERMA_TRANSFORMS_CANONICALIZER_H
#define KERMA_TRANSFORMS_CANONICALIZER_H

#include <llvm/Pass.h>
#include <llvm/PassAnalysisSupport.h>

namespace kerma {

/// This pass is just an aggregator for all
/// Canonicalization passes
class CanonicalizerPass : public llvm::ModulePass {
public:
  static char ID;
  CanonicalizerPass();

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool runOnModule(llvm::Module& M) override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_CANONICALIZER_H