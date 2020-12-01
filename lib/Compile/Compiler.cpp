#include "kerma/Compile/Compiler.h"
#include "boost/filesystem/operations.hpp"
#include "boost/process.hpp"
#include "kerma/Support/FileSystem.h"
#include "kerma/Support/Log.h"

#include <boost/filesystem.hpp>
#include <boost/process/search_path.hpp>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/LLVM.h>
#include <clang/Driver/Driver.h>
// #include <clang/Frontend/CompilerInvocation.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Job.h>
#include <llvm-10/llvm/Support/Program.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>

namespace kerma {

namespace fs = boost::filesystem;
using namespace clang;
using namespace clang::driver;
using namespace llvm;
using namespace boost::process;

Compiler::Compiler(const std::string &ClangPath)
    : ClangPath(ClangPath), DiagOptions(new clang::DiagnosticOptions()),
      DiagPrinter(llvm::errs(), DiagOptions.get()),
      DiagIDs(new clang::DiagnosticIDs()),
      // It is important to pass false here to avoid double-free errors
      Diags(DiagIDs.get(), DiagOptions.get(), &DiagPrinter, false),
      Driver(ClangPath, llvm::sys::getDefaultTargetTriple(), Diags) {
  DiagOptions = new clang::DiagnosticOptions();
}

// -working-directory<arg>, -working-directory=<arg>¶
// Resolve file paths relative to the specified directory
// https://fdiv.net/2012/08/15/compiling-code-clang-api
// https://github.com/llvm/llvm-project/blob/e7fe125b776bf08d95e60ff3354a5c836218a0e6/clang/lib/Driver/ToolChains/Cuda.cpp

// TODO: Use the Clang API to skip writting files
// std::unique_ptr<CompilerInvocation> CI =
// std::make_unique<CompilerInvocation>();
// CompilerInvocation::CreateFromArgs(*CI, getDeviceIRArgs(ClangPath,
// SourcePath), Diags); driver::Driver Driver(ClangPath,
// llvm::sys::getDefaultTargetTriple(), Diags);

const std::string Compiler::DefaultDeviceIRFile = "device.ll";
const std::string Compiler::DefaultHostIRFile = "host.ll";
const std::string Compiler::DefaultDeviceBCFile = "device.bc";
const std::string Compiler::DefaultHostBCFile = "host.bc";
const std::string Compiler::DefaultPTXFile = "device.ptx";
const std::string Compiler::DefaultDeviceObjFile = "device.o";
const std::string Compiler::DefaultFatbinFile = "device.fatbin";
const std::string Compiler::DefaultHostCuiFile = "host.cui";
const std::string Compiler::DefaultLinkedHostIR = "host.linked.ll";

/// The arg lists of these struct are incomplete.
/// They must be prepented by clang binary path
/// and source file path

const std::vector<const char *> BaseArgsCuda = {
    "-g",
    "-O0",
    "-std=c++11",
    "-emit-llvm",
    "--cuda-gpu-arch=sm_30",
    "-Xclang",
    "-disable-O0-optnone",
    "-fno-discard-value-names",
};

static auto GetEmitDeviceIRArgs = [](const std::string &ClangPath,
                                     const std::string &SourcePath,
                                     const std::string &Out) {
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

static auto GetEmitDeviceBCArgs = [](const std::string &ClangPath,
                                     const std::string &SourcePath,
                                     const std::string &Out) {
  std::vector<const char *> Res(BaseArgsCuda);
  Res.push_back("--cuda-device-only");
  Res.push_back("-o");
  Res.push_back(Out.c_str());
  Res.insert(Res.begin(), SourcePath.c_str());
  Res.insert(Res.begin(), ClangPath.c_str());
  return Res;
};

static auto GetEmitHostIRArgs = [](const std::string &ClangPath,
                                   const std::string &SourcePath,
                                   const std::string &Out) {
  std::vector<const char *> Res;
  Res.push_back("-S");
  Res.push_back("--cuda-host-only");
  Res.push_back("-o");
  Res.push_back(Out.c_str());
  Res.insert(Res.begin(), BaseArgsCuda.begin(), BaseArgsCuda.end());
  Res.insert(Res.begin(), SourcePath.c_str());
  Res.insert(Res.begin(), ClangPath.c_str());
  return Res;
};

static auto GetEmitHostCUIArgs = [](const std::string &ClangPath,
                                    const std::string &SourcePath,
                                    const std::string &Out) {
  std::vector<const char *> Res;
  Res.push_back(ClangPath.c_str());
  Res.push_back("-E");
  Res.push_back("--cuda-host-only");
  Res.push_back(SourcePath.c_str());
  Res.push_back("-o");
  Res.push_back(Out.c_str());
  return Res;
};

static bool RunClang(clang::driver::Driver &Driver,
                     const std::vector<const char *> &Args) {
  // FIXME: This is probably a leak. Check whether the clang Drive
  //        automatically deallocates its Compilation when destroyed
  auto *C = Driver.BuildCompilation(Args);
  if (!C)
    return false;
  int Res = 0;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  Res = Driver.ExecuteCompilation(*C, FailingCommands);
  return Res >= 0;
}

bool Compiler::EmitDeviceIR(const std::string &SourcePath,
                            const std::string &Out) {
  auto Arguments = GetEmitDeviceIRArgs(ClangPath, SourcePath, Out);
  // std::unique_ptr<Compilation> Compilation(
  // Driver.BuildCompilation(Arguments)); if ( !Compilation)
  //   return false;
  // int Res = 0;
  // SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  // Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  // return Res >= 0;
  return RunClang(Driver, Arguments);
}

bool Compiler::EmitDeviceBC(const std::string &SourcePath,
                            const std::string &Out) {
  auto Arguments = GetEmitDeviceBCArgs(ClangPath, SourcePath, Out);
  // std::unique_ptr<Compilation> Compilation(
  // Driver.BuildCompilation(Arguments)); if ( !Compilation)
  //   return false;
  // int Res = 0;
  // SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  // Res = Driver.ExecuteCompilation(*Compilation, FailingCommands);
  // return Res >= 0;
  return RunClang(Driver, Arguments);
}

bool Compiler::EmitHostIR(const std::string &SourcePath,
                          const std::string &Out) {
  auto Arguments = GetEmitHostIRArgs(ClangPath, SourcePath, Out);
  return RunClang(Driver, Arguments);
}

bool Compiler::EmitHostBC(const std::string &SourcePath,
                          const std::string &Out) {
  llvm::errs() << "***WARNING*** EmitHostBC() is not implemented!\n";
  return false;
}

bool Compiler::EmitPTX(const std::string &SourcePath, const std::string &Out) {
  auto llc = search_path("llc");
  if (llc.empty()) {
    errs() << "***WARNING*** Could not find llc executable!\n";
    return false;
  } else {
    child c(llc, args({"--march=nvptx64", SourcePath, "--mcpu=sm_30",
                       "-mattr=+ptx60", "--filetype=asm", "-o", Out}));
    c.wait();
    return true;
  }
}

bool Compiler::EmitDeviceObj(const std::string &SourcePath,
                             const std::string &Out) {
  auto ptxas = search_path("ptxas");
  if (ptxas.empty()) {
    errs() << "***WARNING*** Could not find ptxas executable!\n";
    return false;
  } else {
    child c(ptxas, args({"--gpu-name", "sm_30", SourcePath, "-o", Out}));
    c.wait();
    return true;
  }
}

bool Compiler::EmitFatbin(const std::string &SourcePathDeviceObj,
                          const std::string &SourcePathPtx,
                          const std::string &Out) {
  auto fatbin = search_path("fatbinary");
  if (fatbin.empty()) {
    errs() << "***WARNING*** Could not find fatbinary executable!\n";
    return false;
  } else {
    std::string imgprofsm = "--image=profile=sm_30,file=" + SourcePathDeviceObj;
    std::string imgprofcp = "--image=profile=compute_30,file=" + SourcePathPtx;
    child c(fatbin, args({"-64", "--create", Out, imgprofsm, imgprofcp}));
    c.wait();
    return true;
  }
}

} // namespace kerma