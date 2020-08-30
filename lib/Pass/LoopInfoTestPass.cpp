#include "kerma/Pass/LoopInfoTestPass.h"
#include "kerma/Pass/DetectKernels.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"

using namespace llvm;

//https://stackoverflow.com/questions/33356876/try-to-use-llvm-looppass-to-find-number-of-loops-in-program

namespace kerma {

char LoopInfoTestPass::ID = 2;

LoopInfoTestPass::LoopInfoTestPass()
: llvm::ModulePass(ID)
{}

void handleLoop(Loop *L, int i) {
  llvm::errs() << "Loop " << i << "\n\n" << L->getLoopID() << "\n";
}


unsigned countLoops(LoopInfo &LI) {
  unsigned loopCount = 0;
  for ( auto *Loop : LI)
    loopCount++;
  return loopCount;
}

bool LoopInfoTestPass::runOnModule(llvm::Module &M) {

  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();

  unsigned loopcounter = 0;

  for (Function *F : Kernels) {
    if (!F->isIntrinsic() && !F->empty()) {
      LoopInfo& li = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
      
      unsigned loopCount = countLoops(li);

      // if ( !loopCount) continue;

      errs() << "Found " << loopCount << " loop(s) in function " << F->getName() << "\n";
      

      // li.print(errs());

      // for ( auto *Loop : li) {
      //   errs() << LI->getLocRange().getStart().getLine() << "\n";
      // }

    }
    // auto &LI = LIWP.getLoopInfo();

    // for (auto LIT = LI.begin(), LEND = LI.end(); LIT != LEND; ++LIT, ++loopcounter) {
    //   errs() << "YES" << "\n";
    //   handleLoop(*LIT, loopcounter);
    // }
  }
  // DEBUG(errs() << "Found " << loopcounter << " loops.\n");
  return false;
};

void LoopInfoTestPass::print(llvm::raw_ostream &O, const llvm::Module *M) const {

}

void LoopInfoTestPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<DetectKernelsPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesAll();
}


namespace {

static RegisterPass<LoopInfoTestPass> RegisterLoopInfoTestPass(/* pass arg  */   "kerma-li-test", 
                                                               /* pass name */   "Loop Info test pass", 
                                                               /* modifies CFG */ false, 
                                                               /* analysis pass*/ true);
}


} // end namespace kerma
