#include "kerma/Support/Version.h"
#include "kerma/Support/FileSystem.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include <cstdio>
#include <ctime>
#include <exception>
#include <stdexcept>
#include <string>

#if LLVM_VERSION_MAJOR < 9
  #error LLVM version >= 9 is required
#endif

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/Process.h"

#include "clang/Basic/Version.h"

#include "KermadOptions.h"
#include "KermadServer.h"

#include <cstdlib>
#include <iostream>
#include <unistd.h>


/// https://stackoverflow.com/questions/47361538/use-clang-tidy-on-cuda-source-files
/// https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/src/main.cpp
/// https://github.com/llvm/llvm-project/blob/master/clang-tools-extra/clangd/tool/ClangdMain.cpp

using namespace llvm;

namespace kerma {
namespace kermad {
namespace {

cl::OptionCategory GeneralOptions("kermad options");
cl::opt<std::string> OptLog("log"); // FIXME

cl::opt<std::string> OptDirectory("d", 
  cl::cat(GeneralOptions),
  cl::desc("Specify a working directory. By default one is created at the same directory as the executable"),
  cl::value_desc("dir"));

cl::opt<bool> OptPrettyPrint("pretty", 
  cl::cat(GeneralOptions), 
  cl::desc("Pretty pretty JSON output"));


cl::OptionCategory ConnectionOptions("kermad connection options");

cl::opt<unsigned> OptPort("p", 
  cl::cat(ConnectionOptions),
  cl::desc("Specify the port to listen at. "
           "A random available port is picked otherwise"),
  cl::init(0));

const cl::OptionCategory *KermadCategories[] = {&GeneralOptions, &ConnectionOptions};

}

void configureLLVMAndClang(KermadOptions &options) {
  if ( (options.LLVMPath = std::string(std::getenv(options.LLVMEnv.c_str()))).empty() )
    throw std::runtime_error(std::string("Could not find env var $") + options.LLVMEnv);

  options.ClangLibPath = options.LLVMPath + "/lib/clang" + "/" + getLLVMVersion();
  options.ClangLibIncludePath = options.ClangLibPath + "/include";

  if ( !kerma::directoryExists(options.ClangLibIncludePath))
    throw std::runtime_error(std::string("Could not find ") + options.ClangLibIncludePath);
  
  if ( kerma::isEmpty(options.ClangLibIncludePath))
    throw std::runtime_error(options.ClangLibIncludePath + " is empty");
}

} // namespace kermad
} // namespace kerma

int main(int argc, const char** argv) {

  using namespace kerma::kermad;

  KermadOptions Options;

  Options.LLVMEnv = "LLVM_HOME";
  Options.FlagsEnv = "KERMAD_FLAGS";

  const char *Overview =
    R"(kermad is the the Kerma server that allows to interface with the Kerma functionality.

It should be invoked via a frontend tool (such as Kermav) rather than invoked directly. 
For more information, see:
    https://github.com/gkarlos/kerma-view

kermad accepts flags on the commandline and in the KERMAD_FLAGS environment variable.
    )";

  /// Set up CLI
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &OS){
    OS << "kermad version " << kerma::getVersion() << "\n"
       << "llvm version " << LLVM_VERSION_STRING << "\n";
  });

  llvm::cl::HideUnrelatedOptions(KermadCategories);
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview, nullptr, Options.FlagsEnv.c_str());
  llvm::errs().SetBuffered(); // stream can cause significant (non-deterministic) latency for the logger.

  Options.PID          = getpid();
  Options.InvocationId = std::to_string(std::time(0)) + "-" + std::to_string(Options.PID);
  Options.WorkingDir   = OptDirectory.empty()? kerma::get_cwd() : OptDirectory;
  Options.TmpDir       = std::string("tmp-") + Options.InvocationId;
  Options.Port         = OptPort;

  try {
    configureLLVMAndClang(Options);

    KermadServer server(Options);

    server.start();

    dumpKermadOptionsJSON(Options);
  } catch ( const std::runtime_error& e) {
    errs() << e.what() << "\n";
    return 1;
  } catch ( const std::exception& e) {
    errs() << e.what() << "\n";
    return 1;
  } catch ( ... ) {
    errs() << "Unknown error occured. Exiting..." << "\n";
    return 1;
  }
  // std::cout << std::getenv("LLVM_HOME2")).size() << "\n";
  return 0;
}