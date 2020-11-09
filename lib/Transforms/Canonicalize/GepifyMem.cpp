#include "kerma/Transforms/Canonicalize/GepifyMem.h"

#include "kerma/NVVM/NVVMUtilities.h"

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Operator.h>
#include <llvm/Pass.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace kerma {

char GepifyMemPass::ID = 112;
GepifyMemPass::GepifyMemPass() : llvm::FunctionPass(ID) {}

static bool isGEP(Value* V) {
  return dyn_cast_or_null<GetElementPtrInst>(V) != nullptr;
}

// static unsigned int gepify(Instruction *I) {

//   unsigned int changes = 0;

//   if ( !isa<LoadInst>(I) && !isa<StoreInst>(I) && !isa<AtomicRMWInst>(I) )
//     return changes;

//   Value *Op;

//   for ( unsigned int i = 0; i < I->getNumOperands(); ++i ) {
//     Op = I->getOperand(i);

//     // Not sure this check is needed
//     if (!isa<Instruction>(Op) && !isa<Argument>(Op)
//                               && !isa<GlobalVariable>(Op))
//       continue;

//     if ( !isa<ConstantExpr>(Op) && !isa<GetElementPtrInst>(Op)) { // not an inline GEP operator
//       PointerType *PtrTy = dyn_cast<PointerType>(Op->getType());
//       if ( !PtrTy)
//         continue;
//       Type *PointeeTy = PtrTy->getElementType();
//       Value *Zero = ConstantInt::get(IntegerType::getInt64Ty(I->getContext()), 0);
//       auto *GEP = GetElementPtrInst::CreateInBounds(PointeeTy, Op, {Zero}, "gepify", I);
//       GEP->setMetadata("dbg", I->getMetadata("dbg"));
//       I->setOperand(i, GEP);
//       ++changes;
//     }
//   }

//   return changes;
// }

static bool GepifyInstruction(Instruction *I, Value *Ptr) {
  if ( I && Ptr) {
    assert(Ptr->getType()->isPointerTy() && "Pointer value is not PointerType");

    Type *PointeeTy = dyn_cast<PointerType>(Ptr->getType())->getElementType();
    Value *Zero = ConstantInt::get(IntegerType::getInt64Ty(I->getContext()), 0);
    auto *GEP = GetElementPtrInst::CreateInBounds(PointeeTy, Ptr, {Zero}, "gepify", I);
    GEP->setMetadata("dbg", I->getMetadata("dbg"));
    I->replaceUsesOfWith(Ptr, GEP);
    return true;
  }
  errs() << "(warn) <Gepify> Failed on: " << *I << '\n';
  return false;
}


static unsigned int Gepify(Instruction *I) {

  bool Changes = 0;

  if ( auto *MemCpy = dyn_cast<MemCpyInst>(I)) {
    Changes += GepifyInstruction(MemCpy, MemCpy->getRawSource());
    Changes += GepifyInstruction(MemCpy, MemCpy->getRawDest());
  }

  Value *Ptr = nullptr;
  if ( auto *LI = dyn_cast<LoadInst>(I))
    Ptr = LI->getPointerOperand();
  else if ( auto *SI = dyn_cast<StoreInst>(I))
    Ptr = SI->getPointerOperand();
  else if ( auto *CI = dyn_cast<CallInst>(I)) {
    if ( nvvm::isAtomicFunction(*CI->getCalledFunction())
      || nvvm::isReadOnlyCacheFunction(*CI->getCalledFunction()))
    Ptr = CI->getArgOperand(0);
  }

  if ( Ptr)
    Changes += GepifyInstruction(I, Ptr);

  return Changes;
}
  // return Changes;ign 4 addrspacecast (i8 addrspace(4)* bitcast (%struct.Node addrspace(4)* @cnode to i8 addrspace(4)*) to i8*), i64 12, i1 false), !dbg !1121
	// 		@cnode = dso_local addrspace(4) externally_initialized global %st

//   if ( !isa<LoadInst>(I)   &&
//        !isa<StoreInst>(I)  &&
//        !isa<MemCpyInst>(I) &&
//        !(isa<CallInst>(I)  && ( nvvm::isAtomicFunction(*dyn_cast<CallInst>(I)->getCalledFunction())
//                              || nvvm::isReadOnlyCacheFunction(*dyn_cast<CallInst>(I)->getCalledFunction()))
//         ))
//     return 0;

//   errs() << "Gepifying: " << *I << '\n';

//   Value *Op;
//   for ( unsigned int i = 0; i < I->getNumOperands(); ++i) {
//     Op = I->getOperand(i);
//     if ( auto *Cast = dyn_cast<ConstantExpr>(Op)) {
//       llvm::errs() << Cast->getOpcode() << " - " << *Op << " - " << isa<AddrSpaceCastInst>(Cast) << "\n";
//     }


//     if (!isa<Instruction>(Op) && !isa<Argument>(Op)
//                               && !isa<GlobalVariable>(Op))
//       continue;



//     if ( !isa<GEPOperator>(Op) && !isa<GetElementPtrInst>(Op)) {
//       PointerType *PtrTy = dyn_cast<PointerType>(Op->getType());
//       if ( !PtrTy) {
//         errs() << "Skiping: " << *Op << "\n";
//         continue;
//       }
//       Type *PointeeTy = PtrTy->getElementType();
//       Value *Zero = ConstantInt::get(IntegerType::getInt64Ty(I->getContext()), 0);
//       auto *GEP = GetElementPtrInst::CreateInBounds(PointeeTy, Op, {Zero}, "gepify", I);
//       GEP->setMetadata("dbg", I->getMetadata("dbg"));
//       I->setOperand(i, GEP);
//       ++Changes;
//     }
//   }

//   return Changes;
// }

bool GepifyMemPass::runOnFunction(Function &F) {
  if ( F.isDeclaration() || F.isIntrinsic())
    return false;

  unsigned int Changes = 0;

  for ( auto& BB : F) {
    for ( auto& I : BB) {
      Changes += Gepify(&I);
    }
  }

#ifdef KERMA_OPT_PLUGIN
  WithColor::note();
  WithColor(errs(), raw_ostream::Colors::CYAN) << "Gepify: ";
  errs() << demangle(F.getName()) << ": " << Changes << '\n';
#endif
  return Changes;
}

namespace {
static RegisterPass<GepifyMemPass> RegisterGepifyMemPass(
        /* pass arg  */   "kerma-gepify-mem",
        /* pass name */   "Normalize non-gep loads and stores",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}


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