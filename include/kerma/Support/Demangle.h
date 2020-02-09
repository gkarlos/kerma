#ifndef KERMA_STATIC_ANALYSIS_SUPPORT_DEMANGLE_H
#define KERMA_STATIC_ANALYSIS_SUPPORT_DEMANGLE_H

#include <llvm/IR/Function.h>

namespace kerma
{

std::string demangleFn(llvm::Function *f);

std::string demangleFnWithoutArgs(llvm::Function *f);

} /// NAMESPACE kerma

#endif /// KERMA_STATIC_ANALYSIS_SUPPORT_DEMANGLE_H