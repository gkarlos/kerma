#ifndef KERMA_NVVM_NVVM_UTILITIES_H
#define KERMA_NVVM_NVVM_UTILITIES_H

#include "kerma/NVVM/NVVM.h"
#include "llvm/IR/Function.h"

namespace kerma {
namespace nvvm {

/// Check if a function is a CUDA Kernel
bool isKernelFunction(const llvm::Function &F);

/// Check if a function is an NVVM intrinsic
bool isNVVMIntrinsic(const llvm::Function &F);

/// Check if a function is an NVVM atomic intrinsic
bool isNVVMAtomic(const llvm::Function &F);

bool isCudaAPIFunction(const llvm::Function &F);

/// Retrieve the address space corresponding to an ID
/// Defaults to AddressSpace::Unknown;
const AddressSpace::Ty& getAddressSpaceWithId(int id);

/// Check if the name names a CUDA atomic function
bool isAtomic(const std::string& F);

/// Check if the name names a CUDA intrinsic function
bool isIntrinsic(const std::string& F);

bool isNVVMSymbol(const std::string& Symbol);

} // namespace nvvm
} // namespace kerma

#endif // KERMA_NVVM_NVVM_UTILITIES_H
