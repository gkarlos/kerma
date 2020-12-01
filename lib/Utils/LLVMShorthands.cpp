#include "kerma/Utils/LLVMShorthands.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>

using namespace llvm;

namespace kerma {

Type* stripPointers(Type *Ty) {
  Type *tmp = Ty;
  while ( tmp->isPointerTy())
    tmp = dyn_cast<PointerType>(tmp)->getElementType();
  return tmp;
}

unsigned int getPointerDepth(PointerType& PtrTy) {
  unsigned int res = 1;
  llvm::Type *tmp = PtrTy.getElementType();
  while (auto *ptr = dyn_cast<llvm::PointerType>(tmp)) {
    ++res;
    tmp = ptr->getElementType();
  }
  return res;
}

bool isNestedPointer(PointerType& PtrTy) {
  return getPointerDepth(PtrTy) > 1;
}

bool isPointerToStruct(Type *Ty) {
  return Ty && isa<PointerType>(Ty) &&
         dyn_cast<PointerType>(Ty)->getElementType()->isStructTy();
}

bool isArrayOfStructs(Type *Ty) {
  return Ty && isa<ArrayType>(Ty) &&
         dyn_cast<ArrayType>(Ty)->getElementType()->isStructTy();
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

ConstantInt *CreateUnsignedInt(LLVMContext &context, unsigned int value, unsigned numbits) {
  auto *ty = IntegerType::get(context, numbits);
  return ConstantInt::get(ty, value, false);
}

ConstantInt *CreateSignedInt(LLVMContext &context, unsigned int value, unsigned numbits) {
  auto *ty = IntegerType::get(context, numbits);
  return ConstantInt::get(ty, value, true);
}

GlobalVariable *insertGlobalStr(Module &M, llvm::StringRef Str) {
  static unsigned int counter = 0;

  auto* CharTy = IntegerType::get(M.getContext(), 8);

  std::vector<Constant*> chars(Str.size());
  for ( unsigned int i = 0; i < Str.size(); ++i)
    chars[i] = ConstantInt::get(CharTy, Str[i]);
  chars.push_back( ConstantInt::get(CharTy, 0));

  auto* StrTy = ArrayType::get(CharTy, chars.size());

  auto *G = M.getOrInsertGlobal(std::string("arr") + std::to_string(counter++), StrTy);

  if ( G) {
    if ( auto* GV = dyn_cast<GlobalVariable>(G)) {
      GV->setInitializer(ConstantArray::get(StrTy, chars));
      GV->setConstant(true);
      GV->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
      GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      return GV;
    }
  }
  return nullptr;
}

} // end namespace kerma