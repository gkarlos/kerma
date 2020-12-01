#include "kerma/Transforms/Canonicalize/StripPrintf.h"
#include "kerma/Analysis/DetectKernels.h"
#include <llvm-10/llvm/ADT/SmallSet.h>
#include <llvm-10/llvm/Demangle/Demangle.h>
#include <llvm-10/llvm/IR/Instructions.h>

namespace kerma {

using namespace llvm;

char StripPrintfPass::ID = 48;

StripPrintfPass::StripPrintfPass(KernelInfo &KI) : ModulePass(ID), KI(KI) {}

bool
StripPrintfPass::runOnModule(llvm::Module &M) {
  bool Changed = false;

  for ( auto &K : KI.getKernels()) {
    SmallSet<CallInst *, 32> RemoveSet;
    for ( auto &BB : *K.getFunction()) {
      for ( auto &I : BB) {
        if ( auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          StringRef DemangledCalleeName(demangle(Callee->getName()));
          if ( DemangledCalleeName.contains("printf") || DemangledCalleeName.contains("vprintf")) {
            RemoveSet.insert(CI);
          }
        }
      }
    }
    for ( auto *CI : RemoveSet){
      CI->eraseFromParent();
    }
  }
  return Changed;
}

} // namespace kerma