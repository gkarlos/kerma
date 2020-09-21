#include "kerma/Transforms/NormLoadsAndStores.h"

#include <llvm/Pass.h>
#include <llvm/Support/Casting.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace kerma {

char NormLoadsAndStoresPass::ID = 6;

NormLoadsAndStoresPass::NormLoadsAndStoresPass() : llvm::FunctionPass(ID) {}

static bool isGEP(Value* V) {
  return dyn_cast_or_null<GetElementPtrInst>(V) != nullptr;
}

bool NormLoadsAndStoresPass::runOnFunction(Function &F) {
  bool changed = false;

  for ( auto& BB : F) {
    for ( auto& I : BB) {
      if ( !isa<LoadInst>(I) && !isa<StoreInst>(I))
        continue;

      Value *Ptr;

      if ( auto *LI = dyn_cast<LoadInst>(&I))
        Ptr = LI->getPointerOperand();
      else if ( auto *SI = dyn_cast<StoreInst>(&I))
        Ptr = SI->getPointerOperand();
      else
        llvm_unreachable("Load/Store inst but cast failed");

      if ( !isa<GetElementPtrInst>(Ptr)) {
        PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
        Type *PointeeTy = PtrTy->getElementType();
        Value *Zero = ConstantInt::get(IntegerType::getInt64Ty(F.getContext()), 0);
        auto *GEP = GetElementPtrInst::CreateInBounds(PointeeTy, Ptr, {Zero}, "norm", &I);
        GEP->setMetadata("dbg", I.getMetadata("dbg"));
        I.setOperand(0, GEP);
        changed = true;
      }
    }
  }
  return changed;
}


static RegisterPass<kerma::NormLoadsAndStoresPass> RegisterNormLoadsAndStoresPass(
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