
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "spdlog/common.h"
#include "spdlog/pattern_formatter.h"
#include "spdlog/spdlog.h"
#include "llvm/Config/llvm-config.h"
#if LLVM_VERSION_MAJOR < 9
  #error LLVM version >= 9 is required
#endif

#include "clang/Basic/Version.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/VersionTuple.h>

#include "kerma/Support/FileSystem.h"
#include "kerma/Support/Log.h"
#include "kerma/Support/Version.h"
#include "Options.h"
#include "Server.h"
#include "Util.h"

#include <boost/filesystem.hpp>

#include <cstdio>
#include <ctime>
#include <exception>
#include <stdexcept>
#include <string>
#include <signal.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <signal.h>



/// https://stackoverflow.com/questions/47361538/use-clang-tidy-on-cuda-source-files
/// https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/src/main.cpp
/// https://github.com/llvm/llvm-project/blob/master/clang-tools-extra/clangd/tool/ClangdMain.cpp

namespace fs = boost::filesystem;

using namespace kerma;
using namespace kerma::kermad;
using namespace llvm;

cl::OptionCategory GeneralOptions("kermad options");
cl::opt<bool> OptPreserve("preserve",
  cl::cat(GeneralOptions),
  cl::desc("Preserve generated files and directories"),
  cl::init(false));
// cl::opt<std::string> OptDirectory("d",
//   cl::cat(GeneralOptions),
//   cl::desc("Specify a working directory. By default one is created at the same directory as the executable"),
//   cl::value_desc("dir"));

cl::OptionCategory ConnectionOptions("kermad connection options");
cl::opt<unsigned> OptPort("p",
  cl::cat(ConnectionOptions),
  cl::desc("Specify the port to listen at. "
           "A random available port is picked otherwise."
           "(Currently ignored) "),
  cl::init(0));

const cl::OptionCategory *KermadCategories[] = {&GeneralOptions, &ConnectionOptions};
const auto VersionPrinter = [](llvm::raw_ostream &OS) {
    OS << "kermad version " << kerma::getVersion() << "\n"
       << "llvm version " << LLVM_VERSION_STRING << "\n";
  };
const char *Overview =
    R"(kermad is the the Kerma server that allows to interface with the Kerma functionality.

It should be invoked via a frontend tool (such as Kermav) rather than invoked directly.
For more information, see:
    https://github.com/gkarlos/kerma-view

kermad accepts flags on the commandline and in the KERMAD_FLAGS environment variable.
    )";

void configureLLVMAndClang(Options &Options) {
  Options.LLVMEnv = Options.Default.LLVMEnv;
  Options.LLVMPath = std::getenv(Options.LLVMEnv.c_str());

  if ( (Options.LLVMPath = std::string(std::getenv(Options.LLVMEnv.c_str()))).empty() )
    throw std::runtime_error(std::string("Could not find env var $") + Options.LLVMEnv);

  Options.ClangLibPath = Options.LLVMPath + "/lib/clang/10.0.0";
  Options.ClangLibIncludePath = Options.ClangLibPath + "/include";

  if ( !kerma::directoryExists(Options.ClangLibIncludePath))
    throw std::runtime_error(std::string("Could not find ") + Options.ClangLibIncludePath);

  if ( kerma::isEmpty(Options.ClangLibIncludePath))
    throw std::runtime_error(Options.ClangLibIncludePath + " is empty");

  auto P = llvm::sys::findProgramByName("clang++");
  if ( P->empty())
    throw std::runtime_error("Could not locate Clang executable");
  Options.ClangExePath = P.get();
}

void shutdown(Server& Server) {
  Server.stop();
  llvm::llvm_shutdown();
}

void configure(int argc, const char **argv, Options &Options) {
  configureLLVMAndClang(Options);

  Options.InvocationID = "kerma-"+getTimestamp();
  Options.FlagsEnv = Options.Default.FlagsEnv;
  Options.IP = Options.Default.IP;
  Options.Port = Options.Default.Port;
  Options.PID = getpid();

  Options.ExeDir = fs::canonical(fs::system_complete(fs::path( argv[0]))).parent_path().string();
  Options.WorkingDir = Options.Default.WorkingDir;

  // Set up CLI
  llvm::cl::SetVersionPrinter(VersionPrinter);
  llvm::cl::HideUnrelatedOptions(KermadCategories);
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview, nullptr, Options.FlagsEnv.c_str());
  llvm::errs().SetBuffered(); // stream can cause significant (non-deterministic) latency for the logger.

  Log::set_pattern("[%H:%M:%S.%e][%^%L%$] %v");
  Log::set_level(Log::level::debug);

  Log::info("Invocation id: {} (pid: {})", Options.InvocationID, Options.PID);
  Log::info("Listening on: {}:{}", Options.IP, Options.Port);
  Log::info("Using clang executable: {}", Options.ClangExePath);
}


struct Options Opts;

void cleanup() {
  Log::warn("Caught CTRL-C (SIGINT). Cleaning up...");
  if ( !OptPreserve.getValue())
    for ( auto& dir : Opts.CleanupDirs) {
      Log::warn("Removing {}", dir);
      fs::remove_all(dir);
    }
  Opts.Server->stop();
  llvm::llvm_shutdown();
}

void die(int s) {
  cleanup();
  std::exit(s);
}

int main(int argc, const char** argv) {
  signal(SIGINT, die);
  configure(argc, argv, Opts);
  Server Server(Opts);

  Opts.Server = &Server;

  try { Server.start(); }
  catch ( const std::exception& E ) { Log::error(E.what()); }
  catch ( const std::string& E) { Log::error(E); }
  catch ( const std::runtime_error& E) { Log::error(E.what()); }
  catch ( const std::ios::failure& F) { Log::error(F.what()); }
  cleanup();
  return 0;
}