#ifndef KERMA_TRANSFORMS_CANONICALIZER_H
#define KERMA_TRANSFORMS_CANONICALIZER_H

#include <llvm/Pass.h>
#include <llvm/PassAnalysisSupport.h>

namespace kerma {

/// This pass is just an aggregator for all
/// Canonicalization passes
class CanonicalizerPass : public llvm::FunctionPass {
public:
  static char ID;
  CanonicalizerPass();

  // void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool runOnFunction(llvm::Function& F) override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_CANONICALIZER_H