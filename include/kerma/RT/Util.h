#ifndef KERMA_RT_UTIL_H
#define KERMA_RT_UTIL_H

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <stdexcept>

namespace kerma {

enum AccessType : unsigned char {
  Load   = 'L',
  Store  = 'S',
  Atomic = 'A',
  Memcpy = 'C',
  Any    = '*'
};

extern const std::string KermaTraceStatusSymbol;
extern const std::string KermaGlobalSymbolPrefix;
extern const std::string KermaDeviceRTLinkedSymbol;

extern const unsigned int KermaDeviceRTLinkedSymbolValue;
extern const llvm::Type::TypeID KermaDeviceRTLinkedSymbolTypeID;
extern const unsigned int KermaDeviceRTLinkedSymbolTypeSize;

extern const llvm::Type::TypeID KermaTraceStatusSymbolTypeID;
extern const unsigned int KermaTraceStatusSymbolTypeSize;

extern const std::vector<std::string> DeviceRTFunctions;

std::string getAccessTypeAsString(AccessType AT);

class KermaRTNotFoundError : public std::runtime_error {
public:
  KermaRTNotFoundError(const std::string& Msg="");
};

class KermaRTIRParseError : public std::runtime_error {
public:
  KermaRTIRParseError(const std::string& Msg="");
};

/// Check if DeviceRT is linked with a module
bool isDeviceRTLinked(const llvm::Module& M);

/// Check if a function is a DeviceRT function
bool isDeviceRTFunction(std::string& FName);
bool isDeviceRTFunction(llvm::Function& F);

} // namespace kerma

#endif // KERMA_RT_UTIL_H