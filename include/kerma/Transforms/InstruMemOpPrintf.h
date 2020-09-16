#ifndef KERMA_TRANSFORMS_INSTRU_MEM_OP_PRINTF_H
#define KERMA_TRANSFORMS_INSTRU_MEM_OP_PRINTF_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include <stdexcept>

namespace kerma {

class InstruMemOpPrintfPass : public llvm::ModulePass {
public:
  enum Op { Load=1, Store, All };

private:
  std::vector<std::string> Targets;
  Op TargetOp;
  bool IgnoreLocal;

public:
  static char ID;
  InstruMemOpPrintfPass(enum InstruMemOpPrintfPass::Op Op=All, bool IgnoreLocal=false);
  InstruMemOpPrintfPass(const std::vector<std::string>& Targets, bool IgnoreLocal=false);
  InstruMemOpPrintfPass(const std::vector<std::string>& Targets, enum InstruMemOpPrintfPass::Op Op=All, bool IgnoreLocal=false);

public:
  bool runOnModule(llvm::Module &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool hasTargetFunction();
  Op getTargetOp();
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_INSTRU_MEM_OP_PRINTF_H