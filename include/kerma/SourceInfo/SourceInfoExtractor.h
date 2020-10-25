#ifndef KERMA_SOURCEINFO_SOURCEINFO_EXTRACTOR_H
#define KERMA_SOURCEINFO_SOURCEINFO_EXTRACTOR_H

#include "kerma/SourceInfo/FunctionRangeExtractor.h"
#include "kerma/SourceInfo/StmtRangeExtractor.h"
#include "kerma/SourceInfo/SourceRange.h"

#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/ArrayRef.h>
#include <string>
#include <vector>

namespace kerma {

/// This class is a facade for all the lib SourceInfo functionality.
/// In particular it is the preferred way to retrieve source info.
/// A SourceInfoExtractor is associated with a single source file.
/// All methods in the API will trigger (re-)parsing of the source
/// file, because the file may have changed since the last time,
/// and the class makes no attempt to check this. Thus methods
/// should be used with caution.
class SourceInfoExtractor {
private:
  std::unique_ptr<FunctionRangeExtractor> FunRangeExtractor;
  std::string SourcePath;
  clang::tooling::CompilationDatabase* CompileDB;


public:
  SourceInfoExtractor( std::string srcPath,
                       clang::tooling::CompilationDatabase *db=nullptr);

  /// Get the ranges of all functions in the file
  const FunctionRangeExtractor::Result & getAllFunctionRanges();

  /// Get the ranges of a subset of the functions in the file
  /// Missing functions have no entry in the results
  const FunctionRangeExtractor::Result & getFunctionRanges(const std::vector<std::string>& Functions);

  /// Get the range of the function. If the function is overloaded
  /// all ranges are retrieved
  const FunctionRangeExtractor::Result & getFunctionRange(const std::string& Function);

  const StmtRangeRes& getAllStmtRanges();
  const StmtRangeRes& getStmtRangesForFunctions(const std::vector<std::string>& Functions);
  const StmtRangeRes& getStmtRangesForFunction(std::string& Function);

};

};

#endif // KERMA_SOURCEINFO_SOURCEINFO_EXTRACTOR_H

//https://opensource.apple.com/source/clang/clang-425.0.24/src/tools/clang/docs/RAVFrontendAction.html
