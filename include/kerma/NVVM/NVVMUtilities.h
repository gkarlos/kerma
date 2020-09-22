#ifndef KERMA_NVVM_NVVM_UTILITIES_H
#define KERMA_NVVM_NVVM_UTILITIES_H

#include "llvm/IR/Function.h"

namespace kerma {
namespace nvvm {

/// Check if a function is a CUDA Kernel
bool isKernelFunction(const llvm::Function &F);

/// Check if a function is an NVVM intrinsic
bool isNVVMIntrinsic(const llvm::Function &F);

/// Check if a function is an NVVM atomic intrinsic
bool isNVVMAtomic(const llvm::Function &F);

} // namespace nvvm
} // namespace kerma

#endif
