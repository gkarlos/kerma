#ifndef KERMA_TOOLS_KERMAD_KERMADOPTIONS_H
#define KERMA_TOOLS_KERMAD_KERMADOPTIONS_H

#include <memory>

namespace kerma {
namespace kermad {

struct KermadOptions {
  std::string InvocationId;
  std::string WorkingDir;
  std::string TmpDir;
  std::string FlagsEnv;
  std::string LLVMEnv;
  std::string LLVMPath;
  std::string ClangLibPath;
  std::string ClangLibIncludePath;
  unsigned PID;
  unsigned Port;
};

void dumpKermadOptionsJSON(KermadOptions &options);

} // end namespace kermad
} // end namespace kerma

#endif