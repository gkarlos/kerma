#ifndef KERMA_TRANSFORMS_CANONICALIZE_SIMPLIFY_GEP_H
#define KERMA_TRANSFORMS_CANONICALIZE_SIMPLIFY_GEP_H

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/PassAnalysisSupport.h>
#include <llvm/Transforms/InstCombine/InstCombineWorklist.h>

/// This transformation checks for chains of GEP
/// instructions and attempts to merge them

namespace kerma {

/// This pass performs SimplifyGEP on each
/// GetElementPtrInst instruction until no changes
/// occur
class SimplifyGEPPass : public llvm::FunctionPass {
private:
  llvm::SetVector<llvm::Instruction*> DeleteSet;
  bool simplifyGEP(llvm::GetElementPtrInst *GEP);

public:
  static char ID;
  static const char* PASS_NAME;
  SimplifyGEPPass();
  bool runOnFunction(llvm::Function& F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_CANONICALIZE_SIMPLIFY_GEP_H