#include "kerma/Transforms/PrepareMetaDriver.h"
#include <llvm/IR/Attributes.h>

namespace kerma {

using namespace llvm;

char PrepareMetaDriverPass::ID = 50;

PrepareMetaDriverPass::PrepareMetaDriverPass() : ModulePass(ID) {}

bool PrepareMetaDriverPass::runOnModule(llvm::Module &M) {
  unsigned Changes = 0;
  for ( auto &F : M) {
    if ( F.hasFnAttribute(Attribute::OptimizeNone)) {
      Changes++;
      F.removeFnAttr(Attribute::OptimizeNone);
    }
  }
  return Changes;
}

void PrepareMetaDriverPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

} // namespace kerma