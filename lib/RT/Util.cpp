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

const std::string KermaRTLinkedSymbol = "__kerma_rt_linked__";
const unsigned int KermaRTLinkSymbolValue = 0xFEEDC0DE;
const Type::TypeID KermaRTLinkSymbolTypeID = Type::IntegerTyID;

static std::string KermaGlobalNamePrefix = "__kerma_sym";

static std::vector<std::string> DeviceRTFunctions = {
  "__kerma_rec_kernel",
  "__kerma_rec_base",
  "__kerma_rec_access_b",
  "__kerma_rec_access_w",
  "__kerma_rec_access_b_t" };

static std::vector<std::string> DeviceRTFunctionSignatures = {
  "__kerma_rec_kernel(unsigned int, char const*)",
  "__kerma_rec_base(unsigned int, char const*, unsigned int, unsigned int)",
  "__kerma_rec_access_b(unsigned int, unsigned int, unsigned int, unsigned int, char const*, unsigned int)",
  "__kerma_rec_access_w(unsigned int, unsigned int, unsigned int, unsigned int, char const*, unsigned int)",
  "__kerma_rec_access_b_t(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, char const*, unsigned int)" };

/// Check if libKermaRT is linked with a  module
bool KermaRTLinked(const llvm::Module &M) {
  auto &Globals = M.getGlobalList();
  for (auto &global : Globals) {
    if (global.getName() == KermaRTLinkedSymbol) {
      if (global.getInitializer()->getType()->getTypeID() ==
          KermaRTLinkSymbolTypeID) {
        if (auto *ConstInt =
                llvm::dyn_cast<ConstantInt>(global.getInitializer())) {
          if (!(KermaRTLinkSymbolValue ^ ConstInt->getZExtValue())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool isDeviceRTFunction(std::string& FName) {
  return std::find(DeviceRTFunctions.begin(),
                   DeviceRTFunctions.end(),
                   FName) != DeviceRTFunctions.end() ||
         std::find(DeviceRTFunctionSignatures.begin(),
                   DeviceRTFunctionSignatures.end(),
                   FName) != DeviceRTFunctionSignatures.end();
}

bool isDeviceRTFunction(llvm::Function &F) {
  auto name = demangle(F.getName().str());
  return std::find(DeviceRTFunctionSignatures.begin(),
                   DeviceRTFunctionSignatures.end(),
                   name) != DeviceRTFunctionSignatures.end();
}

///
/// Exceptions
///

KermaRTNotFoundError::KermaRTNotFoundError(const std::string &Msg)
    : std::runtime_error("KermaRT not found: " + Msg) {}

KermaRTIRParseError::KermaRTIRParseError(const std::string &Msg)
    : std::runtime_error("KermaRT not parsed: " + Msg) {}

} // namespace kerma