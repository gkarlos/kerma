#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include <llvm/Pass.h>

using namespace llvm;

class SourceInfoPass : public llvm::ModulePass {

public:
  static char ID;

public:
  SourceInfoPass() : llvm::ModulePass(ID)
  {}

  void kernelSourceRange(llvm::Function &F) {

  }

  bool runOnModule(llvm::Module &M) {
    for ( auto &F : M) {
      llvm::errs() << F.getName() << "\n\n";
      for ( auto &BB : F) {
        for ( auto &I : BB) {
          if ( LoadInst *Load = llvm::dyn_cast<llvm::LoadInst>(&I)) {
            llvm::errs() << "Load " << *Load << " | ";
            if ( Load->getDebugLoc())
              llvm::errs() << "at " << Load->getDebugLoc().getLine() << ":" << Load->getDebugLoc().getCol() << "\n";
            else
              llvm::errs() << "unknown location\n";
          } else if ( StoreInst *Store = llvm::dyn_cast<llvm::StoreInst>(&I)) {
            llvm::errs() << "Store " << *Store << " | ";
            if ( Store->getDebugLoc())
              llvm::errs() << "at " << Store->getDebugLoc().getLine() << ":" << Store->getDebugLoc().getCol() << "\n";
            else
              llvm::errs() << "unknown location\n";
          }
        }
      }
      llvm::errs() << "=============================================" << "\n";
    }
    return false;
  }
};

char SourceInfoPass::ID = 1;
static llvm::RegisterPass<SourceInfoPass> X("kerma-source-info", "Source Info Playground", false, true);