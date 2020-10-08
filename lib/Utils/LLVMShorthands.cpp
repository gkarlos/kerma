#include "kerma/Utils/LLVMShorthands.h"
#include <llvm/IR/Constants.h>
#include <llvm/Support/Casting.h>

using namespace llvm;

namespace kerma {

Type* stripPointers(Type *Ty) {
  Type *tmp = Ty;
  while ( tmp->isPointerTy())
    tmp = dyn_cast<PointerType>(tmp)->getElementType();
  return tmp;
}

unsigned int getPointerDepth(llvm::PointerType& PtrTy) {
  unsigned int res = 1;
  llvm::Type *tmp = PtrTy.getElementType();
  while (auto *ptr = dyn_cast<llvm::PointerType>(tmp)) {
    ++res;
    tmp = ptr->getElementType();
  }
  return res;
}

bool isNestedPointer(llvm::PointerType& PtrTy) {
  return getPointerDepth(PtrTy) > 1;
}

std::vector<const llvm::Value *> getGlobalValuesUsedinFunction(const llvm::Function *F) {
  std::vector<const llvm::Value *> GlobalsUsed;
  for (const auto &BB : *F) {
    for (const auto &I : BB) {
      for (const auto &Op : I.operands()) {
        if (const llvm::GlobalValue *G = llvm::dyn_cast<llvm::GlobalValue>(Op)) {
          GlobalsUsed.push_back(G);
        }
        else if ( auto *CE = dyn_cast<ConstantExpr>(Op)) {
          for ( auto& CEOp : CE->operands())
            if (const llvm::GlobalValue *G = llvm::dyn_cast<llvm::GlobalValue>(CEOp)) {
              GlobalsUsed.push_back(G);
            }
        }
      }
    }
  }
  return GlobalsUsed;
}


} // end namespace kerma