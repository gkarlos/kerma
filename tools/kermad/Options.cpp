#include "Options.h"
#include "kerma/Support/FileSystem.h"
#include <llvm/Support/JSON.h>

namespace kerma {
namespace kermad {

const struct DefaultOptions DefaultOpts {
  .WorkingDir = kerma::get_cwd(),
  .FlagsEnv = "KERMAD_FLAGS",
  .LLVMEnv = "LLVM_HOME",
  .IP = "localhost",
  .Port = 8888
};

void dumpKermadOptionsJSON(Options &options) {
  static const char *indent = "  ";

  llvm::errs() << "{\n";

  llvm::errs() << indent << "WorkingDir: " << "\"" << options.WorkingDir << "\",\n";
  llvm::errs() << indent << "PID: " << "\"" << options.PID << "\",\n";
  llvm::errs() << indent << "LLVMEnv: " << "\"" << options.LLVMEnv << "\",\n";
  llvm::errs() << indent << "LLVMPath: " << "\"" << options.LLVMPath << "\",\n";
  llvm::errs() << indent << "ClangLibPath: " << "\"" << options.ClangLibPath << "\",\n";
  llvm::errs() << indent << "ClangLibIncludePath: " << "\"" << options.ClangLibIncludePath << "\",\n";

  llvm::errs() << "}\n";
}

}
}