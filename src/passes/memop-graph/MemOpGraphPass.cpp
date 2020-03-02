//
// Created by gkarlos on 1/8/20.
//
#include <kerma/passes/memop-graph/MemOpGraph.h>
#include <kerma/passes/detect-kernels/DetectKernels.h>
#include <kerma/Support/LLVMStringUtils.h>

#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

using namespace llvm;

namespace kerma {

void
processLoad(LoadInst *LI, MemoryDependenceResults &MDA, int i) {
    errs() << "-- " << i << ". Load @" << LI->getDebugLoc().getLine() << ":" << LI->getDebugLoc().getCol() << " | " << *LI << "\n";
//      auto dep2 = MDA.getDependency(d);
//      if ( auto d2 = dep.getInst())
//        errs() << "  " << *d2 << " = " << (d2 == d) << "\n";
}

void processStore(StoreInst *SI, MemoryDependenceResults &MDA, int i) {
  errs() << "-- " << i << ". Store @";
  if ( SI->getDebugLoc()) {
    errs() << SI->getDebugLoc().getLine() << ":" << SI->getDebugLoc().getCol()
           << *SI << "\n";
  } else
    errs() << "??" << *SI << "\n";

  errs() << "   VAL: " << *SI->getOperand(0) << "\n";
  errs() << "   PTR: " << *SI->getOperand(1) << "\n";
  errs() << "   DEP: ";
  if ( MDA.getDependency(SI).getInst()) {
    // The dependency of a StoreInst seems to always be a LoadInst
    // There may be operations on the loaded value between the
    // LoadInst and the StoreInst
    // TODO: Some StoreInsts do not have dependency:
    //    It looks like a StoreInst that has an AllocaInst target does not
    //    have a dependency
    errs() << *(MDA.getDependency(SI).getInst()) << "\n";
  }
  else
    errs() << " -- \n";
}

void
MemOpGraphPass::analyzeKernel(CudaKernel* kernel)
{
  Function &F = kernel->getFn();
  auto& MSSA = getAnalysis<MemorySSAWrapperPass>(F).getMSSA();

  auto walker = MSSA.getWalker();

  for ( auto &BB : F) {
    for ( auto &I : BB) {
      if ( LoadInst *load = dyn_cast<LoadInst>(&I)) {
        errs() << "[+] Load:" << *load << " - " << getDbgLocString(load) << "\n";
        for ( Use &u : load->operands()) {
          u->users();
        }
      }
      else if ( StoreInst *store = dyn_cast<StoreInst>(&I)) {
        errs() << "[+] Store: " << *store << " - " << getDbgLocString(store) << "\n";
      }
    }
  }

  MSSA.print(errs());

//  for ( auto& BB : *F)
//    for ( auto& I : BB) {
//      Instruction *Inst = &I;
//      if (!Inst->mayReadFromMemory() && !Inst->mayWriteToMemory())
//        continue;
//
//      errs() << *Inst << "\n";
//      MemDepResult Res = MDA.getDependency(Inst);
//      if (!Res.isNonLocal()) {
//        errs() << "  -- Local: " << *Res.getInst() << "\n";
//      } else if (auto *Call = dyn_cast<CallBase>(Inst)) {
//        const MemoryDependenceResults::NonLocalDepInfo &NLDI =
//            MDA.getNonLocalCallDependency(Call);
//        errs() << "  -- Call:\n";
//        for (const NonLocalDepEntry &I : NLDI) {
//          const MemDepResult &Res = I.getResult();
//          errs() << "       " << *Res.getInst() << "\n";
//        }
//      }
//    }
}

bool
MemOpGraphPass::runOnModule(Module &M) {

  auto kernels = getAnalysis<DetectKernelsPass>().getKernels();

  for ( auto* kernel : kernels) {
    MemoryDependenceResults &MDA = getAnalysis<MemoryDependenceWrapperPass>(kernel->getFn()).getMemDep();
    analyzeKernel(kernel);
  }

  return false;
}

void
MemOpGraphPass::print(llvm::raw_ostream &OS, const llvm::Module *M) const {
  // TODO
}

void
MemOpGraphPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
  AU.addRequiredTransitive<AAResultsWrapperPass>();
  AU.addRequired<MemoryDependenceWrapperPass>();
  AU.addRequired<MemorySSAWrapperPass>();
}

}

char kerma::MemOpGraphPass::ID = 0;
static RegisterPass<kerma::MemOpGraphPass> Y("kerma-memop-graph", "Build Memory Dependency Graph", false, true);
