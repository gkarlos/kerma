#ifndef KERMA_STATIC_ANALYSIS_DETECTADDRESSSPACE_H
#define KERMA_STATIC_ANALYSIS_DETECTADDRESSSPACE_H


#include "kerma/Cuda/NVVM.h"
#include <llvm/Pass.h>
#include <llvm/IR/Value.h>

namespace kerma
{

class DetectAddrSpacePass : public llvm::ModulePass {

public:
  static char ID;
  DetectAddrSpacePass() : llvm::ModulePass(ID)
  {}

  bool runOnModule(llvm::Module &M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  // API

  AddressSpace getAddrSpace(llvm::Value *v);

  bool maybeGlobal(llvm::Value *v);

  bool provablyGlobal(llvm::Value *v);
};

}

#endif /* KERMA_STATIC_ANALYSIS_DETECTADDRESSSPACE_H */