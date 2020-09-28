#ifndef KERMA_ANAYLSIS_INFER_ADDRESS_SPACES_H
#define KERMA_ANAYLSIS_INFER_ADDRESS_SPACES_H

#include "kerma/NVVM/NVVM.h"
#include "llvm/IR/Value.h"

namespace kerma {

/// Clear the cache
void clearAddressSpaceCache();

/// Removes an entry if it exists and returns true.
/// Returns false otherwise
bool clearAddressSpaceCacheEntry(llvm::Value *V);

/// Insert/Update a value in the cache.
/// Returns true if the value was inserted.
/// Returns false if the value was present but updated.
bool cacheAddressSpaceForValue(llvm::Value *V, const nvvm::AddressSpace::Ty& AS);

/// Retrieve the address space of a __global__ or __device__
/// function argument. The address space returned is the where
/// the argument itself is copied.
/// To get the address space of the memory a ptr arg points to,
/// use getAddressSpaceForArg()
const nvvm::AddressSpace::Ty& getAddressSpaceForArgValue(llvm::Argument& Arg, bool isKernelArg);

/// Try to infer the address space of a value
/// Defaults to AddressSpace::Generic
/// Returns AddressSpace::Unknown if something went wrong
const nvvm::AddressSpace::Ty& getAddressSpace(llvm::Value *V, bool kernel=true);

} // namespace kerma

#endif // KERMA_ANAYLSIS_INFER_ADDRESS_SPACES_H