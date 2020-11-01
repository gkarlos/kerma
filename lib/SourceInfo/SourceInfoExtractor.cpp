#include "kerma/SourceInfo/SourceInfoExtractor.h"
#include "kerma/SourceInfo/FunctionRangeExtractor.h"
#include "kerma/Support/FileSystem.h"

#include "llvm/Support/FileSystem.h"

#include <clang/Tooling/CompilationDatabase.h>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace kerma {

using namespace llvm;
using namespace clang;

// https://github.com/ROCm-Developer-Tools/HIPIFY/blob/c43543b88cd7693add3c6ecfb4387fde4198eb35/src/main.cpp#L126
// https://github.com/ROCm-Developer-Tools/HIPIFY/blob/c43543b88cd7693add3c6ecfb4387fde4198eb35/src/main.cpp#L219
// https://github.com/ROCm-Developer-Tools/HIPIFY/blob/c43543b88cd7693add3c6ecfb4387fde4198eb35/src/main.cpp#L364

SourceInfoExtractor::SourceInfoExtractor( std::string SourcePath,
                                          clang::tooling::CompilationDatabase *DB)
: CompileDB(DB) {
  if ( !sys::fs::exists(SourcePath))
    throw SourcePath + " does not exist";
  if ( sys::fs::is_directory(SourcePath))
    throw SourcePath + " is a directory";

  if ( !DB) {
    std::string Err = "Could not read compile_commands.json";
    auto db = tooling::CompilationDatabase::autoDetectFromSource(SourcePath,Err);
    if ( !db)
      throw Err;

    // Look only under the same directory as SourcePath
    // auto Parent = llvm::sys::path::parent_path(SourcePath);
    // auto CompileCommandsFile = Parent + "/compile_commands.json";
    // if ( !sys::fs::exists(CompileCommandsFile))
    //   throw std::runtime_error(CompileCommandsFile.str() + " not found");
    // auto db = tooling::CompilationDatabase::loadFromDirectory(Parent, Err);

    CompileDB = db.release();
  }

  auto RealSourcePath = get_realpath(SourcePath);
  if ( RealSourcePath.empty())
    throw std::runtime_error(std::string("Error getting real path for ") + SourcePath);
  /// The requested file must have an entry in the Compilation Database
  if ( auto DBFiles = CompileDB->getAllFiles();
       std::find(DBFiles.begin(), DBFiles.end(), RealSourcePath) == DBFiles.end())
    throw std::runtime_error(RealSourcePath + " not found in CompilationDatabase");

  this->SourcePath = RealSourcePath;

  this->FunRangeExtractor = std::make_unique<FunctionRangeExtractor>(RealSourcePath, CompileDB);
}

const FunctionRangeExtractor::Result & SourceInfoExtractor::getAllFunctionRanges() {
  return FunRangeExtractor->getFunctionRanges();
}

const FunctionRangeExtractor::Result & SourceInfoExtractor::getFunctionRanges(const std::vector<std::string>& Functions) {
  return FunRangeExtractor->getFunctionRanges(Functions);
}

const FunctionRangeExtractor::Result & SourceInfoExtractor::getFunctionRange(const std::string& Function) {
  return FunRangeExtractor->getFunctionRange(Function);
}


};