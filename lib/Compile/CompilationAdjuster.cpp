#include "kerma/Compile/CompilationAdjuster.h"

#include "llvm/Support/FileSystem.h"

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"

#include <cstdlib>
#include <iostream>

namespace kerma {

// TODO These values are probably needed elsewhere too. So better
//      move them to some sort of config file
const char * CompilationAdjuster::AssumedLLVMVersion = "10";
const char * CompilationAdjuster::AssumedClangVersion = "10.0.0";
const char * CompilationAdjuster::ExpectedLLVMHomeEnvVar = "LLVM_HOME";
const char * CompilationAdjuster::ExpectedClangIncludeEnvVar = "CLANG_INC";

static std::string ArgIncludePre = "-I";

void CompilationAdjuster::appendCommandLineArg( clang::tooling::ClangTool &tool, const char *argument) {
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(argument, clang::tooling::ArgumentInsertPosition::END));
}

void CompilationAdjuster::prependCommandLineArg( clang::tooling::ClangTool &tool, const char *argument) {
  tool.appendArgumentsAdjuster( clang::tooling::getInsertArgumentAdjuster(argument, clang::tooling::ArgumentInsertPosition::BEGIN));
}

bool CompilationAdjuster::appendClangIncludes(clang::tooling::ClangTool& tool) {
  std::string ArgClangIncludes = ArgIncludePre;
  std::string ArgCudaWrappers = ArgIncludePre;

  if (auto *ClangIncludes = std::getenv(CompilationAdjuster::ExpectedClangIncludeEnvVar)) {
    if ( llvm::sys::fs::exists(ClangIncludes) && llvm::sys::fs::is_directory(ClangIncludes)) {
      ArgClangIncludes += ClangIncludes;
      ArgCudaWrappers += std::string(ClangIncludes) + "/cuda_wrappers";
      appendCommandLineArg(tool, ArgClangIncludes.c_str());
      appendCommandLineArg(tool, ArgCudaWrappers.c_str());
      return true;
    }
  }

  if ( auto *LLVMHome = std::getenv(CompilationAdjuster::ExpectedLLVMHomeEnvVar)) {
    if ( llvm::sys::fs::exists(LLVMHome)) {
      std::string ClangIncludes = std::string(LLVMHome) + "/lib/clang/" + AssumedClangVersion + "/include";
      if ( llvm::sys::fs::exists(ClangIncludes) && llvm::sys::fs::is_directory(ClangIncludes)) {
        ArgClangIncludes += ClangIncludes;
        ArgCudaWrappers += ClangIncludes + "/cuda_wrappers";

        appendCommandLineArg(tool, ArgClangIncludes.c_str());
        appendCommandLineArg(tool, ArgCudaWrappers.c_str());
        appendCommandLineArg(tool, "-xcuda");
        // appendCommandLineArg(tool, "-v", clang::tooling::ArgumentInsertPosition::END);
        return true;
      }
    }
  }

  return false;
}

} // namespace kerma