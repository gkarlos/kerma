#include "kerma/Analysis/InferDimensions.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace kerma {


unsigned int getPtrDimensions(PointerType *PtrTy, bool ByValue) {
  auto* PointeeTy = PtrTy->getElementType();

  if ( auto* ArrTy = dyn_cast<ArrayType>(PointeeTy))
    return ByValue? getArrayDimensions(ArrTy, ByValue) : (1 + getArrayDimensions(ArrTy, ByValue));
  if ( auto *StructTy = dyn_cast<StructType>(PointeeTy))
    return ByValue? getTypeDimensions(StructTy, ByValue) : (1 + getTypeDimensions(StructTy, ByValue));

  unsigned int dims = ByValue? 0 : 1;
  while ( isa<PointerType>(PointeeTy)) {
    ++dims;
    PointeeTy = dyn_cast<PointerType>(PointeeTy)->getElementType();
  }

  return dims;
}

unsigned int getArrayDimensions(ArrayType *ArrTy, bool ArgByValOrAlloca) {
  return ArrTy->getNumContainedTypes()? (1 + getTypeDimensions(ArrTy->getContainedType(0))) : 1;
}

unsigned int getTypeDimensions(Type *Ty, bool ByVal) {
  if ( auto *PtrTy = dyn_cast<PointerType>(Ty)) {
    return getPtrDimensions(PtrTy, ByVal);
  }
  else if ( auto *StructTy = dyn_cast<StructType>(Ty)) {
    return 0;
  }else if ( auto *ArrTy = dyn_cast<ArrayType>(Ty)){
    return getArrayDimensions(ArrTy);
  } else
    return 0;
}

unsigned int getNumDimensions(Value *V) {
  if ( auto *Arg = dyn_cast<Argument>(V)) {
    auto res = getTypeDimensions(Arg->getType(), Arg->hasAttribute(Attribute::ByVal));
    llvm::errs() << "arg: " << *Arg << " - " << res << "-D\n";
    return res;
  } else if ( auto* GV = dyn_cast<GlobalVariable>(V)) {
    auto res = getTypeDimensions(GV->getType(), true);
    llvm::errs() << "gv: " << *GV << " - " << res << "-D\n";
    return res;
  } else if ( auto* Alloca = dyn_cast<AllocaInst>(V)) {
    auto res = getTypeDimensions(Alloca->getType(), true);
    llvm::errs() << "alloca: " << *Alloca << " - " << res << "-D";
    return res;
  } else if ( auto* LI = dyn_cast<LoadInst>(V)) {
    auto res = getNumDimensions(LI->getPointerOperand());
    llvm::errs() << "load: " << *LI << " - " << res << "-D\n";
  } else if ( auto* GEP = dyn_cast<GetElementPtrInst>(V))
    return getNumDimensions(GEP->getPointerOperand());
  return 0;
}

} // namespace kerma