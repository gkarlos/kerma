#pragma once

#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/JSONCompilationDatabase.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include <memory>
#include <string>

namespace kerma {

/// This class provides functionality to retrieve additional source info
/// from IR instructions. It helps to associate IR values with source code 
/// ranges, in particular end lines and columns, which is currently not
/// available through LLVM APIs
class SourceInfo {

static std::string COMPILEDB_READ_ERR_MSG;

public:
  SourceInfo(llvm::Module *M, const char *CompileDBFilePath) 
  : SourceInfo(M, std::string(CompileDBFilePath))
  {}

  SourceInfo(llvm::Module *M, std::string &&CompileDBFilePath)
  : SourceInfo(M, clang::tooling::JSONCompilationDatabase::loadFromFile(CompileDBFilePath, 
                                                                        SourceInfo::COMPILEDB_READ_ERR_MSG,
                                                                        clang::tooling::JSONCommandLineSyntax::AutoDetect).release())
  {}

  SourceInfo(llvm::Module *M, clang::tooling::CompilationDatabase *compileDB);
 
public:
  /// Get the the of the source file
  std::string getFilename();

  /// Get the directory of the source file
  std::string getDirectory();

  /// Get the full path of the source file
  std::string getFullPath();

  /// Get the source file contents
  std::string getSource();

  /// Get the source range of a function.
  /// If no source range info is found, the returned SourceRange will be invalid. 
  /// @See SourceRange::isValid() and SourceRange::isInvalid()
  clang::SourceRange getFunctionRange(const char *);

  void testFunc(const char *file);

  // /// Get the source range of the source statement an instruction is part of. 
  // /// If no source range info is found, the returned SourceRange will be invalid. 
  // /// @See SourceRange::isValid() and SourceRange::isInvalid()
  // clang::SourceRange getInstructionStmtRange(llvm::Instruction *I);

  // /// Get the source range of the target of a memory write
  // clang::SourceRange getStoreLhs(llvm::StoreInst *SI);

  // // Get the source range of the source of a memory write
  // clang::SourceRange getStoreRhs(llvm::StoreInst *SI);

  // // Get the number of statements that a line
  // // overlaps with. In most cases this will be 0 or 1
  // unsigned int getNumStatementsAtLine(int lineno);


private:
  llvm::Module *M;
  llvm::DICompileUnit *CompileUnitMD;
  clang::tooling::CompilationDatabase *CompileDB;
  std::string Src;

private:
  void init();
};
 
} // namespace kerma
