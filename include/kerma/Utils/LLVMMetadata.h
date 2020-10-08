#ifndef KERMA_UTILS_LLVM_METADATA_H
#define KERMA_UTILS_LLVM_METADATA_H

#include <llvm/IR/Argument.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>

namespace kerma {

/// Get the metadata for a function argument
llvm::DILocalVariable *findMDForArgument(llvm::Argument *Arg);

/// Get metadata for a local variable
llvm::DILocalVariable *findMDForAlloca(llvm::AllocaInst *AI);

/// Get metadata for a global variable
llvm::DIGlobalVariable *findMDForGlobal(llvm::GlobalVariable *GV);


void clearCache();
void clearCache(llvm::Function *F);
void clearLocalCache();
void clearLocalCache(llvm::Function *F);

} // end namespace kerma

#endif // KERMA_UTILS_LLVM_METADATA_H