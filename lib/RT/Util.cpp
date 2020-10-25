#include "kerma/RT/Util.h"

#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <llvm/ADT/APInt.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

using namespace llvm;
namespace kerma {

const std::string KermaTraceStatusSymbol    = "__kerma_trace_status__";
const std::string KermaGlobalSymbolPrefix   = "__kerma_sym";
const std::string KermaDeviceRTLinkedSymbol = "__kerma_rt_linked__";

const unsigned int KermaDeviceRTLinkedSymbolValue    = 0xFEEDC0DE;
const Type::TypeID KermaDeviceRTLinkedSymbolTypeID   = Type::IntegerTyID;
const unsigned int KermaDeviceRTLinkedSymbolTypeSize = 4;

const Type::TypeID KermaTraceStatusSymbolTypeID   = Type::IntegerTyID;
const unsigned int KermaTraceStatusSymbolTypeSize = 1;

const std::vector<std::string> DeviceRTFunctions = {
  "__kerma_trace_status",
  "__kerma_stop_tracing",
  "__kerma_rec_kernel",
  "__kerma_rec_base",
  "__kerma_rec_access_b",
  "__kerma_rec_access_w",
  "__kerma_rec_access_t",
  "__kerma_rec_copy_b",
  "__kerma_rec_copy_w",
  "__kerma_rec_copy_t"
};

/// Check if DeviceRT is linked with a  module
bool isDeviceRTLinked(const llvm::Module &M) {
  auto &Globals = M.getGlobalList();
  for (auto &global : Globals) {
    if (global.getName() == KermaDeviceRTLinkedSymbol) {
      if (auto *ty = global.getInitializer()->getType();
          ty->getTypeID() == KermaDeviceRTLinkedSymbolTypeID) {
        if (auto *ConstInt = llvm::dyn_cast<ConstantInt>(global.getInitializer())) {
          if (!(KermaDeviceRTLinkedSymbolValue ^ ConstInt->getZExtValue())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool isDeviceRTFunction(const std::string& FName) {
  auto name = demangle(FName);
  for ( auto &Fn : DeviceRTFunctions)
    if ( name.find(Fn) != name.npos)
      return true;
  return false;
}

bool isDeviceRTFunction(llvm::Function &F) {
  return isDeviceRTFunction(F.getName().str());
}

std::string getAccessTypeAsString(AccessType AT) {
  if ( AT == Load)
    return "Load";
  else if ( AT == Store)
    return "Store";
  else if ( AT == Atomic)
    return "Atomic";
  else if ( AT == Memcpy)
    return "Memcpy";
  else
    return "Any";
}


//
// Exceptions
//

KermaRTNotFoundError::KermaRTNotFoundError(const std::string &Msg)
    : std::runtime_error("KermaRT not found: " + Msg) {}

KermaRTIRParseError::KermaRTIRParseError(const std::string &Msg)
    : std::runtime_error("KermaRT not parsed: " + Msg) {}

} // namespace kerma