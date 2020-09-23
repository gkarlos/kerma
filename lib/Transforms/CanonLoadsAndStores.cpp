#include "kerma/Transforms/CanonLoadsAndStores.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Pass.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace kerma {

char CanonLoadsAndStoresPass::ID = 6;

CanonLoadsAndStoresPass::CanonLoadsAndStoresPass() : llvm::FunctionPass(ID) {}

static bool isGEP(Value* V) {
  return dyn_cast_or_null<GetElementPtrInst>(V) != nullptr;
}

static unsigned int insertGepForLoadStore(Instruction *I) {

  unsigned int changes = 0;

  if ( !isa<LoadInst>(I) && !isa<StoreInst>(I))
    return changes;

  Value *Op;

  for ( unsigned int i = 0; i < I->getNumOperands(); ++i ) {
    Op = I->getOperand(i);

    // Not sure this check is needed
    if (!isa<Instruction>(Op) && !isa<Argument>(Op)
                              && !isa<GlobalVariable>(Op))
      continue;

    if ( !isa<ConstantExpr>(Op) && !isa<GetElementPtrInst>(Op)) { // not an inline GEP operator
      PointerType *PtrTy = dyn_cast<PointerType>(Op->getType());
      if ( !PtrTy)
        continue;
      Type *PointeeTy = PtrTy->getElementType();
      Value *Zero = ConstantInt::get(IntegerType::getInt64Ty(I->getContext()), 0);
      auto *GEP = GetElementPtrInst::CreateInBounds(PointeeTy, Op, {Zero}, "nrm.ls", I);
      GEP->setMetadata("dbg", I->getMetadata("dbg"));
      I->setOperand(i, GEP);
      ++changes;
    }
  }

  return changes;
}

bool CanonLoadsAndStoresPass::runOnFunction(Function &F) {
  bool changed = false;

  for ( auto& BB : F) {
    for ( auto& I : BB) {
      if ( !isa<LoadInst>(I) && !isa<StoreInst>(I))
        continue;
      changed |= insertGepForLoadStore(&I);
    }
  }
  return changed;
}


static RegisterPass<kerma::CanonLoadsAndStoresPass> RegisterCanonLoadsAndStoresPass(
        /* pass arg  */   "kerma-normalize-ls",
        /* pass name */   "Normalize non-gep loads and stores",
        /* modifies CFG */ false,
        /* analysis pass*/ false);

} // namespace kerma

  // with Mystruct.C[0]
  // %7 = getelementptr inbounds %struct.s, %struct.s* %4, i32 0, i32 1, !dbg !1846
  // %8 = load i32*, i32** %7, align 8, !dbg !1846
  // %9 = getelementptr inbounds i32, i32* %8, i64 0, !dbg !1847
  // %10 = load i32, i32* %9, align 4, !dbg !1847

  // with *MyStruct.C
  // %7 = getelementptr inbounds %struct.s, %struct.s* %4, i32 0, i32 1, !dbg !1846
  // %8 = load i32*, i32** %7, align 8, !dbg !1846
  // %norm = getelementptr inbounds i32, i32* %8, i64 0, !dbg !1847
  // %9 = load i32, i32* %norm, align 4, !dbg !1847