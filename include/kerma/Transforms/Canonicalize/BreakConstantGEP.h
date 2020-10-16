#ifndef KERMA_TRANSFORMS_CANONICALIZE_BREAK_CONSTANT_GEP_H
#define KERMA_TRANSFORMS_CANONICALIZE_BREAK_CONSTANT_GEP_H

#include <llvm/Pass.h>

namespace kerma {

class BreakConstantGEPPass : public llvm::FunctionPass {
public:
  static char ID;
  BreakConstantGEPPass();
  bool runOnFunction(llvm::Function &F) override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_CANONICALIZE_BREAK_CONSTANT_GEP_H