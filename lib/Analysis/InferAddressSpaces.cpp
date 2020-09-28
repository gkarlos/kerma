#include "kerma/Analysis/InferAddressSpaces.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/NVVM/NVVM.h"
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/raw_ostream.h>
#include <map>


using namespace llvm;
using namespace kerma::nvvm;

namespace kerma {

static Value* stripCasts(Value *V) {
  auto *Res = V;
  while ( auto *cast = dyn_cast<BitCastOperator>(Res))
    Res = cast->getOperand(0);
  return Res;
}

using AddressSpaceCacheTy = std::map<Value *, AddressSpace::Ty const *>;

//TODO: ManagedStatic
static AddressSpaceCacheTy AddressSpaceCache;


static const AddressSpace::Ty& join(const AddressSpace::Ty& A, const AddressSpace::Ty& B) {
  return A.ID > B.ID? A : B;
}

bool cacheAddressSpaceForValue(llvm::Value *V, const nvvm::AddressSpace::Ty& AS) {
  assert(V && "Cannot insert null in AddressSpaceCache");
  if ( auto entry = AddressSpaceCache.find(V); entry != AddressSpaceCache.end()) {
    entry->second = &AS;
    return false;
  } else {
    AddressSpaceCache[V] = &AS;
    return true;
  }
}

/// Cache the address space for a value and return it
static inline const AddressSpace::Ty& cache(Value *V, const AddressSpace::Ty& AS) {
  cacheAddressSpaceForValue(V, AS);
  return AS;
}

const AddressSpace::Ty& getAddressSpaceForArgValue(Argument& Arg, bool isKernelArg) {
  if ( isKernelArg) {
    // TODO: 1. Kernel argument placement is (in general) arch specific. So far,
    //          for CC >= 2.x, arguments are copied to constant addr space.
    //       2. For CC >= 2.x the argument limit is 4KB. It is unclear where
    //          the arguments are copied it they exceed that. Assume global.
    //
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-parameters
    return AddressSpace::Constant;
  }
  return AddressSpace::Local;
}

/// Infer the address space of a value
///
/// 1. Pointer Arguments:
///      -> get pointee's addr space (always global?)
/// 2. Non pointer arguments:
///      -> usually constant
const AddressSpace::Ty& getAddressSpace(Value *V, bool kernel) {
  if ( !V) // bail out
    return AddressSpace::Unknown;

  if ( auto entry = AddressSpaceCache.find(V); entry != AddressSpaceCache.end())
    return *(entry->second);

  if ( auto* LI = dyn_cast<LoadInst>(V))
    return cache(V, getAddressSpace(LI->getPointerOperand(), kernel));
  if ( auto* SI = dyn_cast<StoreInst>(V))
    return cache(V, getAddressSpace(SI->getPointerOperand(), kernel));
  if ( auto* GEP = dyn_cast<GetElementPtrInst>(V))
    return cache(V, join( getAddressSpaceWithId(GEP->getPointerAddressSpace()),
                          getAddressSpace(GEP->getPointerOperand(), kernel)));
  if ( auto *UO = dyn_cast<UnaryOperator>(V))
    return cache( V, getAddressSpace(UO->getOperand(0), kernel));
  if ( auto *BO = dyn_cast<BinaryOperator>(V))
    return cache( V, join( getAddressSpace(BO->getOperand(0), kernel),
                           getAddressSpace(BO->getOperand(1), kernel)));

  if ( auto* AI = dyn_cast<AllocaInst>(V))
    // stack ptrs are opaque. There is no corresponding alloca
    // in the IR. So all Allocas are stack scalars/aggregates
    return cache( V, AddressSpace::Local);
  if ( auto* Arg = dyn_cast<Argument>(V)) {
    auto* F = Arg->getParent();
    if ( !F)
      // FIX: should this be an assertion?
      return cache( V, AddressSpace::Unknown);
    if ( kernel)
      return isa<PointerType>(Arg->getType())
          ? cache( V, AddressSpace::Global)
          : cache( V, getAddressSpaceForArgValue(*Arg, kernel));
    else
      return cache( V, AddressSpace::Generic);
  }

  if ( auto* CI = dyn_cast<CallInst>(V)) {
    auto* Callee = CI->getCalledFunction();
    if ( !Callee)
      return cache( V, AddressSpace::Generic);
    if ( nvvm::isAtomic(demangle(Callee->getName())))
      return cache( V, getAddressSpace(CI->getOperand(0), kernel));
    if ( Callee->getName().startswith("malloc") )
      return cache( V, AddressSpace::Global);
  }

  if ( auto* GV = dyn_cast<GlobalVariable>(V)) {
    if ( auto* GVTy = GV->stripPointerCasts()->getType())
      return cache( V, nvvm::getAddressSpaceWithId(GVTy->getPointerAddressSpace()));
  }

  return cache(V, AddressSpace::Generic);
}

} // namespace kerma