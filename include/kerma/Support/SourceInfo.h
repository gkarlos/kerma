#pragma once


#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

/// This class provides functionality to retrieve additional source info
/// from IR instructions. It helps to associate IR values with source code 
/// ranges, in particular end lines and columns, which is currently not
/// available through LLVM APIs
class RangeLocator {
private:
  llvm::StringRef file;
public:
  RangeLocator(const char *file) : file(file) 
  {}

public:
  /// Get the file this locator will be searching
  llvm::StringRef getFile();

  /// Get the source range of a function.
  /// If no source range info is found, the returned SourceRange will be invalid. 
  /// @See SourceRange::isValid() and SourceRange::isInvalid()
  clang::SourceRange getFunctionRange(llvm::Function *F);

  /// Get the source range of the source statement an instruction is part of. 
  /// If no source range info is found, the returned SourceRange will be invalid. 
  /// @See SourceRange::isValid() and SourceRange::isInvalid()
  clang::SourceRange getStmtRange(llvm::Instruction *I);

  /// Get the source range of the target of a memory write
  clang::SourceRange getStoreLhs(llvm::StoreInst *SI);

  // Get the source range of the source of a memory write
  clang::SourceRange getStoreRhs(llvm::StoreInst *SI);

  // Get the number of statements that a line
  // overlaps with. In most cases this will be 0 or 1
  int getNumStatementsAtLine(int lineno);

};
