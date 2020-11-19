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
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Host.h>
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
  DiagIDs(new clang::DiagnosticIDs()),
  // It is important to pass false here to avoid double-free errors
  Diags(DiagIDs.get(), DiagOptions.get(), &DiagPrinter, false),
  Driver(ClangPath, llvm::sys::getDefaultTargetTriple(), Diags)
{
  DiagOptions = new clang::DiagnosticOptions();
}

// -working-directory<arg>, -working-directory=<arg>¶
// Resolve file paths relative to the specified directory
// https://fdiv.net/2012/08/15/compiling-code-clang-api
// https://github.com/llvm/llvm-project/blob/e7fe125b776bf08d95e60ff3354a5c836218a0e6/clang/lib/Driver/ToolChains/Cuda.cpp

// TODO: Use the Clang API to skip writting files
// std::unique_ptr<CompilerInvocation> CI = std::make_unique<CompilerInvocation>();
// CompilerInvocation::CreateFromArgs(*CI, getDeviceIRArgs(ClangPath, SourcePath), Diags);
// driver::Driver Driver(ClangPath, llvm::sys::getDefaultTargetTriple(), Diags);

const std::string Compiler::DefaultDeviceIRFile = "device.ll";
const std::string Compiler::DefaultHostIRFile = "host.ll";
const std::string Compiler::DefaultDeviceBCFile = "device.bc";
const std::string Compiler::DefaultHostBCFile = "host.bc";

/// The arg lists of these struct are incomplete.
/// They must be prepented by clang binary path
/// and source file path

const std::vector<const char *> BaseArgsCuda = {
  "-g", "-O0", "-std=c++11", "-emit-llvm", "--cuda-gpu-arch=sm_30",
  "-Xclang", "-disable-O0-optnone", "-fno-discard-value-names",
};

static auto GetEmitDeviceIRArgs =
  [](const std::string& ClangPath, const std::string& SourcePath, const std::string& Out) {
    std::vector<const char *> Res;
    Res.push_back("-S");
    Res.push_back("--cuda-device-only");
    Res.push_back("-o");
    Res.push_back(Out.c_str());
    Res.insert(Res.begin(), BaseArgsCuda.begin(), BaseArgsCuda.end());
    Res.insert(Res.begin(), SourcePath.c_str());
    Res.insert(Res.begin(), ClangPath.c_str());
    return Res;
  };

static auto GetEmitDeviceBCArgs =
  [](const std::string& ClangPath, const std::string& SourcePath, const std::string& Out) {
    std::vector<const char *> Res(BaseArgsCuda);
    Res.push_back("--cuda-device-only");
    Res.push_back("-o");
    Res.push_back(Out.c_str());
    Res.insert(Res.begin(), SourcePath.c_str());
    Res.insert(Res.begin(), ClangPath.c_str());
    return Res;
  };

static bool RunClang(clang::driver::Driver &Driver, const std::vector<const char *> &Args ) {
  std::unique_ptr<Compilation> Compilation( Driver.BuildCompilation(Args));
  if ( !Compilation)
    return false;
  int Res = 0;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  return Res >= 0;
}

bool Compiler::EmitDeviceIR(const std::string& SourcePath, const std::string& Out) {
  auto Arguments = GetEmitDeviceIRArgs(ClangPath, SourcePath, Out);
  // std::unique_ptr<Compilation> Compilation( Driver.BuildCompilation(Arguments));
  // if ( !Compilation)
  //   return false;
  // int Res = 0;
  // SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  // Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  // return Res >= 0;
  return RunClang(Driver, Arguments);
}

bool Compiler::EmitDeviceBC(const std::string& SourcePath, const std::string& Out) {
  auto Arguments = GetEmitDeviceBCArgs(ClangPath, SourcePath, Out);
  // std::unique_ptr<Compilation> Compilation( Driver.BuildCompilation(Arguments));
  // if ( !Compilation)
  //   return false;
  // int Res = 0;
  // SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  // Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  // return Res >= 0;
  return RunClang(Driver, Arguments);
}

bool Compiler::EmitHostIR(const std::string &SourcePath, const std::string& Out) {
  llvm::errs() << "***WARNING*** EmitHostIR() is not implemented!\n";
  return false;
}

bool Compiler::EmitHostBC(const std::string &SourcePath, const std::string& Out) {
  llvm::errs() << "***WARNING*** EmitHostBC() is not implemented!\n";
  return false;
}


} // namespace kerma