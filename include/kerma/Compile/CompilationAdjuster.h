#ifndef KERMA_COMPILE_COMPILE_DB_ADJUSTER_H
#define KERMA_COMPILE_COMPILE_DB_ADJUSTER_H

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"

namespace kerma {

/// Contains functionality, available as static methods
/// to modify Clang or ClangTool invocations
class CompilationAdjuster {
public:
  CompilationAdjuster()=delete;
  static const char * AssumedLLVMVersion;
  static const char * AssumedClangVersion;
  static const char * ExpectedClangIncludeEnvVar;
  static const char * ExpectedLLVMHomeEnvVar;

public:
  /// Look for the Clang includes directory and append to
  /// to the ClangTool's command line
  /// @returns true if the Clang includes are found, otherwise false.
  static bool appendClangIncludes(clang::tooling::ClangTool &tool);

  /// Insert a command line argument to a ClangTool
  /// The argument is inserted at the end
  static void appendCommandLineArg(clang::tooling::ClangTool &tool, const char *argument);

  /// Insert a command line argument to a ClangTool
  /// The argument is inserted at the beginning
  static void prependCommandLineArg(clang::tooling::ClangTool &tool, const char *argument);
};

} // namespace kerma


#endif // KERMA_COMPILE_COMPILE_DB_ADJUSTER_H