#ifndef KERMA_PASS_MATERIALIZE_COMPUTE_UNIT_H
#define KERMA_PASS_MATERIALIZE_COMPUTE_UNIT_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace kerma {

class MaterializeComputeUnit : public llvm::FunctionPass {

public:
  bool runOnFunction(llvm::Function &F) override;
  virtual void print(llvm::raw_ostream &O, const llvm::Module *M) const override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  std::vector<llvm::Function*> kernels;
  
};

} // namespace kerma

#endif