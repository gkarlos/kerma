#include <kerma/Support/Demangle.h>

#include <llvm/Config/llvm-config.h>


#if LLVM_VERSION_MAJOR < 9
#include <cxxabi.h>
#else
#include <llvm/Demangle/Demangle.h>
#endif

namespace kerma {

std::string demangleFn(const llvm::Function &f)
{
#if LLVM_VERSION_MAJOR < 9
  int status;
  char *p = abi::__cxa_demangle( f.getName().str().c_str(), nullptr, nullptr, &status);
  if ( status != 0)
    return std::string(f.getName().str().append("_(could not demangle)"));
  std::string demangled(p);
  free(p);
  return demangled;
#else
  return llvm::demangle(f.getName().str());
#endif
}

std::string demangleFnWithoutArgs(const llvm::Function &f)
{
  std::string demangledName = kerma::demangleFn(f);
  if ( auto argstart = demangledName.find('(')) {
    demangledName = demangledName.substr(0, argstart);
  }
  return demangledName;
}

}



