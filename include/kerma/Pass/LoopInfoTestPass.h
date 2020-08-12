#ifndef KERMA_PASS_LOOP_INFO_TEST_PASS_H
#define KERMA_PASS_LOOP_INFO_TEST_PASS_H

#include "llvm/Pass.h"

namespace kerma {

class LoopInfoTestPass : public llvm::ModulePass {

public:
  static char ID;
  LoopInfoTestPass();

public:
  bool runOnModule(llvm::Module &M) override;
  virtual void print(llvm::raw_ostream &O, const llvm::Module *M) const override;
  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // end namespace kerma

#endif