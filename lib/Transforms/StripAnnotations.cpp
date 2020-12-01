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

/// FIXME: At the moment, this pass only removes calls to
///        llvm.var.annotation, which is fine for the
///        subsequent analysis passes, however it does not
///        remove the global strings created for the
///        annotations. These contribute to the size of the
///        binary. We should also remove those at some point
StripAnnotationsPass::StripAnnotationsPass(KernelInfo &KI)
    : KI(KI), ModulePass(ID) {}


bool StripAnnotations(Function *F) {
  SmallVector<Instruction*, 32> Erase;
  for ( auto &I : F->getEntryBlock()) {
  if ( auto *CI = dyn_cast<CallInst>(&I))
    if ( CI->getCalledFunction()->getName().startswith("llvm.var.annotation"))
      Erase.push_back(CI);
  }
  bool Changed = !Erase.empty();

  while ( !Erase.empty()) {    
    auto *I = Erase.pop_back_val();
    I->replaceAllUsesWith(nullptr);
    I->eraseFromParent();
  }
  return Changed;
}

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