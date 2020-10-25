#include "kerma/Utils/LLVMMetadata.h"

#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>

#include <map>

using namespace llvm;

namespace kerma {

using LocalMetadataCache = std::map<Value*, DILocalVariable*>;

static std::map<Function*, LocalMetadataCache> LocalCache;
static std::map<GlobalVariable*, DIGlobalVariable*> GlobalCache;


DILocalVariable *findMDForArgument(Argument *Arg) {
  if ( !Arg)
    return nullptr;

  if ( auto *Cached = LocalCache[Arg->getParent()][Arg])
    return Cached;

  for ( auto &BB : *Arg->getParent() ) {
    for ( auto &I : BB) {
      if ( auto *DVI = dyn_cast<DbgValueInst>(&I)) {
        if ( auto *op1 = dyn_cast<MetadataAsValue>(DVI->getOperand(1))) {
          if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
            if ( DILV->isParameter()) {
              if ( DILV->getArg() - 1 == Arg->getArgNo()) {
                LocalCache[Arg->getParent()][Arg] = DILV;
                return DILV;
              }
            }
          }
        }
      }
      else if ( auto* DDI = dyn_cast<DbgDeclareInst>(&I)) {
        if ( auto *op1 = dyn_cast<MetadataAsValue>(DDI->getOperand(1)))
          if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
            if ( DILV->isParameter()) {
              if ( DILV->getArg() - 1 == Arg->getArgNo()) {
                LocalCache[Arg->getParent()][Arg] = DILV;
                return DILV;
              }
            }
          }
      }
    }
  }
  return nullptr;
}

DILocalVariable *findMDForAlloca(AllocaInst *AI) {
  if ( !AI)
    return nullptr;

  if ( auto *Cached = LocalCache[AI->getParent()->getParent()][AI])
    return Cached;

  // TODO: Retrieving metadata by checking the name is not robust
  // It will fail for something like this:
  //
  //   int A[42];
  //   ..
  //   if ( ... ) {
  //     float A[42];
  //     ...
  //   }
  //
  // No metadata will be returned for the second Alloca because
  // since both variables have the same name, for the second
  // Alloca, getName() will return something like "A2"
  // but metadata getName() will be "A"
  for ( auto &BB : *AI->getParent()->getParent() ) {
    for ( auto &I : BB) {
      if ( auto *DVI = dyn_cast<DbgValueInst>(&I)) {
        if ( auto *op1 = dyn_cast<MetadataAsValue>(DVI->getOperand(1))) {
          if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
            if ( DILV->getName().equals(AI->getName())) {
              LocalCache[AI->getParent()->getParent()][AI] = DILV;
              return DILV;
            }
          }
        }
      }
      else if ( auto *DDI = dyn_cast<DbgDeclareInst>(&I)) {
        if ( auto *op1 = dyn_cast<MetadataAsValue>(DDI->getOperand(1))) {
          if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
            if ( DILV->getName().equals(AI->getName())) {
              LocalCache[AI->getParent()->getParent()][AI] = DILV;
              return DILV;
            }
          }
        }
      }
    }
  }

  return nullptr;
}

DIGlobalVariable *findMDForGlobal(GlobalVariable *GV) {
  if ( !GV)
    return nullptr;

  if ( auto *Cached = GlobalCache[GV])
    return Cached;

  if ( auto *MD = GV->getMetadata("dbg"))
    if ( auto *DIGVExpr = dyn_cast<DIGlobalVariableExpression>(MD))
      if ( auto *DIGV = DIGVExpr->getVariable()) {
        GlobalCache[GV] = DIGV;
        return DIGV;
      }

  return nullptr;
}

void clearCache() {
  clearLocalCache();
  // TODO: Add the rest of the caches
}

void clearCache(llvm::Function *F) {
  clearLocalCache(F);
  // TODO: Add teh rest of the caches
}

void clearLocalCache() {
  LocalCache.clear();
}


void clearLocalCache(llvm::Function *F) {
  if ( LocalCache.find(F) != LocalCache.end())
    LocalCache[F].clear();
}

} // end namespace kerma
