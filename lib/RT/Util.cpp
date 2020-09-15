#include "kerma/RT/Util.h"

#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

namespace kerma {
namespace rt {

using namespace llvm;

const std::string KermaRTLinkedSymbol      = "___kerma_rt_linked___";
const unsigned int KermaRTLinkSymbolValue  = 0xFEEDC0DE;
const Type::TypeID KermaRTLinkSymbolTypeID = Type::IntegerTyID;

/// Check if libKermaRT is linked with a  module
bool KermaRTLinked(const llvm::Module& M) {
  auto& Globals = M.getGlobalList();
  for ( auto& global : Globals) {
    if ( global.getName() == KermaRTLinkedSymbol) {
      if ( global.getInitializer()->getType()->getTypeID() == KermaRTLinkSymbolTypeID) {
        if ( auto* ConstInt = llvm::dyn_cast<ConstantInt>(global.getInitializer())) {
          if ( !(KermaRTLinkSymbolValue ^ ConstInt->getZExtValue())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}


} // namespace rt
} // namespace kerma