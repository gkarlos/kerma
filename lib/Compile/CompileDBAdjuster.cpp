#include "kerma/Compile/CompileDBAdjuster.h"

#include "llvm/Support/FileSystem.h"
#include <llvm/Support/Path.h>

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"

#include <cstdlib>
#include <iostream>

namespace kerma {

// TODO These values are probably needed elsewhere too. So better
//      move them to some soft of config file
const char * CompileDBAdjuster::AssumedLLVMVersion = "10";
const char * CompileDBAdjuster::AssumedClangVersion = "10.0.0";
const char * CompileDBAdjuster::ExpectedLLVMHomeEnvVar = "LLVM_HOME";
const char * CompileDBAdjuster::ExpectedClangIncludeEnvVar = "CLANG_INC";

static std::string ArgIncludePre = "-I";

namespace {
  void appendCommandLineArg(clang::tooling::ClangTool &tool, const char *argument,
                            clang::tooling::ArgumentInsertPosition pos=clang::tooling::ArgumentInsertPosition::BEGIN) {
    tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(argument, pos));
  }
}

bool CompileDBAdjuster::appendClangIncludes(clang::tooling::ClangTool& tool) {
  std::string ArgClangIncludes = ArgIncludePre;

  if (auto *ClangIncludes = std::getenv(CompileDBAdjuster::ExpectedClangIncludeEnvVar)) {
    if ( llvm::sys::fs::exists(ClangIncludes) && llvm::sys::fs::is_directory(ClangIncludes)) {
      ArgClangIncludes += ClangIncludes;
      appendCommandLineArg(tool, ArgClangIncludes.c_str());
      std::cout << "Using: " << ClangIncludes << "\n";
      return true;
    }
  }
  
  if ( auto *LLVMHome = std::getenv(CompileDBAdjuster::ExpectedLLVMHomeEnvVar)) {
    if ( llvm::sys::fs::exists(LLVMHome)) {
      std::string ClangIncludes = std::string(LLVMHome) + "/lib/clang/" 
                                + CompileDBAdjuster::AssumedClangVersion + "/include";
      if ( llvm::sys::fs::exists(ClangIncludes) && llvm::sys::fs::is_directory(ClangIncludes)) {
        ArgClangIncludes += ClangIncludes;
        appendCommandLineArg(tool, ArgClangIncludes.c_str());
        std::cout << "Using: " << ClangIncludes << "\n";
        return true;
      }
    }
  }
  
  return false;
}

} // namespace kerma