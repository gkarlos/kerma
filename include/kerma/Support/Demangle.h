#ifndef KERMA_SUPPORT_DEMANGLE_H
#define KERMA_SUPPORT_DEMANGLE_H

#include <llvm/IR/Function.h>

namespace kerma
{

std::string demangleFn(const llvm::Function &f);

std::string demangleFnWithoutArgs(const llvm::Function &f);

} /// NAMESPACE kerma

#endif /// KERMA_SUPPORT_DEMANGLE_H