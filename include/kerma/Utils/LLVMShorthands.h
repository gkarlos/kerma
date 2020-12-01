#ifndef KERMA_UTILS_LLVM_SHORTHANDS_H
#define KERMA_UTILS_LLVM_SHORTHANDS_H

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>

namespace kerma {

/// Ignore pointers in the type. Return the
/// the first non-pointer type encountered
llvm::Type* stripPointers(llvm::Type *Ty);

/// Get the number of pointers in the type
/// e.g int*   -> 1
///     int**  -> 2
///     int*** -> 3
unsigned int getPointerDepth(llvm::PointerType& PtrTy);

/// Check if its a ptr to ptr to ... chain
bool isNestedPointer(llvm::PointerType& PtrTy);

bool isPointerToStruct(llvm::Type *Ty);

bool isArrayOfStructs(llvm::Type *Ty);

/// Retrieve a list of globals used in a function
std::vector<const llvm::Value *> getGlobalValuesUsedinFunction(const llvm::Function *F);

///
llvm::GlobalVariable *insertGlobalStr(llvm::Module &M, llvm::StringRef Str);

llvm::ConstantInt *CreateUnsignedInt(llvm::LLVMContext &context, unsigned int value, unsigned numbits);
llvm::ConstantInt *CreateSignedInt(llvm::LLVMContext &context, unsigned int value, unsigned numbits);

} // namespace kerma

#endif // KERMA_UTILS_LLVM_SHORTHANDS_H