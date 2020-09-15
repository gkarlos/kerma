#ifndef KERMA_RT_UTIL_H
#define KERMA_RT_UTIL_H

#include "llvm/IR/Module.h"

namespace kerma {
namespace rt {

/// Check if libKermaRT is linked with a  module
bool KermaRTLinked(const llvm::Module& M);

} // namespace rt
} // namespace kerma

#endif // KERMA_RT_UTIL_H