#include "kerma/Transforms/StripAnnotations.h"
#include "kerma/Analysis/DetectKernels.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Scalar/DCE.h>

namespace kerma {

using namespace llvm;

char StripAnnotationsPass::ID = 5;

StripAnnotationsPass::StripAnnotationsPass(KernelInfo &KI)
    : KI(KI), ModulePass(ID) {}


bool StripAnnotationsPass::runOnModule(llvm::Module &M) {

  bool Changed = false;

  for ( auto &Kernel : KI.getKernels()) {
    SmallVector<Instruction*, 32> Erase;

    for ( auto &I : Kernel.getFunction()->getEntryBlock()) {
      if ( auto *CI = dyn_cast<CallInst>(&I))
        if ( CI->getCalledFunction()->getName().startswith("llvm.var.annotation"))
          Erase.push_back(CI);
    }

    while ( !Erase.empty()) {
      Changed = true;
      auto *I = Erase.pop_back_val();
      I->replaceAllUsesWith(nullptr);
      I->eraseFromParent();
    }
  }

  return Changed;
}

} // namespace kerma