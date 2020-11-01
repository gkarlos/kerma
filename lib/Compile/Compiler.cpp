#include "kerma/Compile/Compiler.h"
#include "boost/filesystem/operations.hpp"
#include "kerma/Support/Log.h"
#include "kerma/Support/FileSystem.h"

#include <boost/filesystem.hpp>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/LLVM.h>
#include <clang/Driver/Driver.h>
// #include <clang/Frontend/CompilerInvocation.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Job.h>
#include <llvm-10/llvm/ADT/ArrayRef.h>
#include <llvm-10/llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace kerma {

namespace fs = boost::filesystem;
using namespace clang;
using namespace clang::driver;
using namespace llvm;

Compiler::Compiler(const std::string& ClangPath)
: ClangPath(ClangPath),
  DiagOptions(new clang::DiagnosticOptions()),
  DiagPrinter(llvm::errs(), DiagOptions.get()),
  DiagID(new clang::DiagnosticIDs()),
  Diags(DiagID, DiagOptions.get(), &DiagPrinter),
  Driver(ClangPath, llvm::sys::getDefaultTargetTriple(), Diags)
{}

// -working-directory<arg>, -working-directory=<arg>¶
// Resolve file paths relative to the specified directory
// https://fdiv.net/2012/08/15/compiling-code-clang-api
// https://github.com/llvm/llvm-project/blob/e7fe125b776bf08d95e60ff3354a5c836218a0e6/clang/lib/Driver/ToolChains/Cuda.cpp

// TODO: Use the Clang API to skip writting files
// std::unique_ptr<CompilerInvocation> CI = std::make_unique<CompilerInvocation>();
// CompilerInvocation::CreateFromArgs(*CI, getDeviceIRArgs(ClangPath, SourcePath), Diags);
// driver::Driver Driver(ClangPath, llvm::sys::getDefaultTargetTriple(), Diags);

// static std::vector<const char *> getDeviceIRArgs(const std::string& ClangPath,
//                                                  const std::string& SourcePath,
//                                                  const std::string& OutputPath) {
//   return {
//     ClangPath.c_str(), SourcePath.c_str(),
//     "-g", "-O0", "-std=c++11", "-S", "-emit-llvm",
//     "--cuda-device-only", "--cuda-gpu-arch=sm_30",
//     "-Xclang", "-disable-O0-optnone", "-fno-discard-value-names"
//     "-o", DEVICE_IR
//   };
// }

/// The arg lists of these struct are incomplete.
/// They must be prepented by clang binary path
/// and source file path
static struct {
  /// These args are used to compile Device IR
  const std::vector<const char *> DeviceIR = {
    "-g", "-O0", "-std=c++11", "-S", "-emit-llvm",
    "--cuda-device-only", "--cuda-gpu-arch=sm_30",
    "-Xclang", "-disable-O0-optnone", "-fno-discard-value-names",
    "-o", DEVICE_IR
  };
} Args;




/// Generate device side LLVM IR.
/// On Support a file named device.ll
/// will be created in the current dir
bool Compiler::getDeviceIR(const std::string& SourcePath) {

  std::vector<const char*> Arguments{ ClangPath.c_str(), SourcePath.c_str()};

  Arguments.insert(Arguments.end(), Args.DeviceIR.begin(), Args.DeviceIR.end());

  std::unique_ptr<Compilation> Compilation( Driver.BuildCompilation(Arguments));

  if ( !Compilation)
    return false;

  int Res = 0;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  Log::info("Compiling {}", SourcePath);
  return Res > 0;
}

} // namespace kerma