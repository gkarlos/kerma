#ifndef KERMA_COMPILE_COMPILE_DB_ADJUSTER_H
#define KERMA_COMPILE_COMPILE_DB_ADJUSTER_H

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"

namespace kerma {

/// Contains functionality, available as static methods
/// to modify Compilation Databases
class CompileDBAdjuster {
public:
  CompileDBAdjuster()=delete;
  static const char * AssumedLLVMVersion;
  static const char * AssumedClangVersion;
  static const char * ExpectedClangIncludeEnvVar;
  static const char * ExpectedLLVMHomeEnvVar;

public:
  /// Look for the Clang includes directory and append to 
  /// to the ClangTool's command line
  static bool appendClangIncludes(clang::tooling::ClangTool &tool);

  /// Append a Command Line Argument to a ClangTool
  // static void appendCLArg(clang::tooling::ClangTool &tool, const char *arg,
  //                         clang::tooling::ArgumentInsertPosition pos=clang::tooling::ArgumentInsertPosition::BEGIN);
};

} // namespace kerma


#endif