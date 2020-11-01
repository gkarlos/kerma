#ifndef KERMA_TOOLS_KERMAD_KERMADOPTIONS_H
#define KERMA_TOOLS_KERMAD_KERMADOPTIONS_H

#include <memory>
#include <set>

namespace kerma {
namespace kermad {

class Server;

struct DefaultOptions {
  std::string WorkingDir;
  std::string FlagsEnv;
  std::string LLVMEnv;
  std::string IP;
  unsigned short int Port;
};

extern const DefaultOptions DefaultOpts;

struct Options {
  /// Directory of the kermad executable
  std::string ExeDir;
  /// Name of the env var used to pass args
  /// to the kermad invocation (default: KERMAD_FLAGS)
  std::string FlagsEnv;
  /// Name of the env var used to locate
  /// LLVM (default: LLVM_HOME)
  std::string LLVMEnv;
  /// Path to the LLVM installation (value of LLVMEnv)
  std::string LLVMPath;
  /// Path to the clang executable
  std::string ClangExePath;
  /// Path to clang's lib dir
  std::string ClangLibPath;
  /// Path to clang's includes
  std::string ClangLibIncludePath;
  /// Working Dir for this kermad invocation
  /// (default: CWD)
  std::string WorkingDir;
  /// An identifier for the current
  /// kermad invocation
  std::string InvocationID;
  /// IP of kermad (default: localhost)
  std::string IP;
  /// Handler to the kermad Server
  Server *Server;
  /// The port kermad is listening on (default: 8888)
  unsigned short int Port;
  /// PID of the kermad invocation
  unsigned short int PID;
  /// List of directories to cleanup when
  /// kermad exits
  std::set<std::string> CleanupDirs;
  /// Default values for (some) options
  const DefaultOptions& Default=DefaultOpts;
  /// Add a directory to the cleanup list
  void addCleanupDir(const std::string& Dir) { CleanupDirs.insert(Dir); }
};

void dumpKermadOptionsJSON(Options &options);

} // end namespace kermad
} // end namespace kerma

#endif