#include "KermadOptions.h"
#include "llvm/Support/JSON.h"

namespace kerma {
namespace kermad {

void dumpKermadOptionsJSON(KermadOptions &options) {
  static const char *indent = "  ";

  llvm::errs() << "{\n";

  llvm::errs() << indent << "InvocationId: " << "\"" << options.InvocationId << "\",\n";
  llvm::errs() << indent << "WorkingDir: " << "\"" << options.WorkingDir << "\",\n";
  llvm::errs() << indent << "TmpDir: " << "\"" << options.TmpDir << "\",\n";
  llvm::errs() << indent << "PID: " << "\"" << options.PID << "\",\n";
  llvm::errs() << indent << "LLVMEnv: " << "\"" << options.LLVMEnv << "\",\n";
  llvm::errs() << indent << "LLVMPath: " << "\"" << options.LLVMPath << "\",\n";
  llvm::errs() << indent << "ClangLibPath: " << "\"" << options.ClangLibPath << "\",\n"; 
  llvm::errs() << indent << "ClangLibIncludePath: " << "\"" << options.ClangLibIncludePath << "\",\n";

  llvm::errs() << "}\n";
}

}
}