#include "kerma/Analysis/Loops.h"
#include <llvm-10/llvm/Transforms/Utils.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/PassSupport.h>
#include <llvm/Analysis/ScalarEvolution.h>


using namespace llvm;

namespace kerma {

char LoopTestPass::ID = 43;

LoopTestPass::LoopTestPass() : FunctionPass(ID) {}

void LoopTestPass::getAnalysisUsage(AnalysisUsage &AU) const {
  // AU.addRequired<PromoteMemoryToRegisterPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

bool LoopTestPass::runOnFunction(Function &F) {
  // auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  llvm::errs() << F.getName() << '\n';
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  for ( auto *L : LI) {
    errs() << L->getLoopDepth() << '\n';
    errs() << L->getLocRange().getStart().getLine() << '\n';
    errs() << L->getNumBackEdges() << '\n';
    errs() << SE.getSmallConstantTripCount(L) << '\n';
  }

  return false;
}

// namespace {

// static RegisterPass<LoopTestPass> RegisterLoopTestPass(
//     /* pass arg  */ "kerma-loop-test",
//     /* pass name */ "Check loops",
//     /* modifies CFG */ false,
//     /* analysis pass*/ true);
// }
} // namespace kerma

