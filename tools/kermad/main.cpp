#include "kerma/Support/Version.h"

#include "llvm/Config/llvm-config.h"

#if LLVM_VERSION_MAJOR < 9
  #error LLVM version >= 9 is required
#endif

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/Process.h"

#include "clang/Basic/Version.h"

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

cl::OptionCategory GeneralOptions("General options");
cl::opt<std::string> Log("log"); // FIXME
cl::opt<bool> PrettyPrint("pretty", cl::cat(GeneralOptions), 
                                    cl::desc("Pretty pretty JSON output"));

cl::OptionCategory ConnectionOptions("kermad connection options");
cl::opt<unsigned> Port("p", cl::cat(ConnectionOptions),
                            cl::desc("Specify the port to listen at. "
                                     "A random available port is picked otherwise"));

const cl::OptionCategory *KermadCategories[] = {&GeneralOptions, &ConnectionOptions};

}
} // namespace kermad
} // namespace kerma

int main(int argc, const char** argv) {

  using namespace kerma;
  using namespace kermad;

  const char *FlagsEnvVar = "KERMAD_FLAGS";
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
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview, nullptr, FlagsEnvVar);

  // If a user just ran `kermad` in a terminal it's somewhat 
  // likely they're confused about how to use kermad.
  // Show them the help overview, which explains.
  // if (llvm::outs().is_displayed() && llvm::errs().is_displayed())
  //   llvm::errs() << Overview << "\n";

  // stream can cause significant (non-deterministic) latency for the logger.
  llvm::errs().SetBuffered();
  
  llvm::errs() << "PID: " << getpid() << "\n";

  // std::cout << std::getenv("LLVM_HOME2")).size() << "\n";
  return 0;
}