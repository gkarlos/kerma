#ifndef KERMA_SUPPORT_LLVMFUNCTIONSHORTHANDS_H
#define KERMA_SUPPORT_LLVMFUNCTIONSHORTHANDS_H

#include <llvm/IR/Function.h>

namespace kerma
{

/*
 * Retrieve the number of arguments in an llvm::Function
 */
int getFnNumArgs(llvm::Function &fn);

}

#endif /* KERMA_SUPPORT_LLVMFUNCTIONSHORTHANDS_H */