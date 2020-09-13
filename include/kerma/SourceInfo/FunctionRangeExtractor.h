#ifndef KERMA_SOURCEINFO_FUNCTION_RANGE_EXTRACTOR_H
#define KERMA_SOURCEINFO_FUNCTION_RANGE_EXTRACTOR_H

#include "kerma/SourceInfo/SourceRange.h"

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include <string>
#include <unordered_map>

namespace kerma {

using FunctionRangeRes = std::unordered_map< std::string,
                                             std::vector<kerma::SourceRange>>;


/// This class provides an API for extracting source code ranges
/// for Functions in a file. Each extractor is associated with a
/// file only and methods trigger an invocation of a ClangTool.
class FunctionRangeExtractor {
private:
  class FunctionRangeActionFactory; // forward declaration
  std::vector<std::string> SourcePaths;
  clang::tooling::CompilationDatabase *CompileDB;
  std::unique_ptr<clang::tooling::ClangTool> Tool;
  std::unique_ptr<FunctionRangeActionFactory> ActionFactory;

private:
  unsigned int runTool() const;

public:
  FunctionRangeExtractor( std::string SourcePath, clang::tooling::CompilationDatabase *DB);

  void getFunctionRanges(FunctionRangeRes& Res);
  void getFunctionRanges(std::vector<std::string>& Targets, FunctionRangeRes &Res);
  void getFunctionRange(const std::string& Target, FunctionRangeRes &Res);

  /// Get the ranges of all functions in the file
  const FunctionRangeRes& getFunctionRanges() const;

  /// Get the ranges of specific function in the file
  /// If Targets is empty, then the ranges of all the
  /// functions in the file are returned
  const FunctionRangeRes& getFunctionRanges(const std::vector<std::string> &Targets);
  const FunctionRangeRes& getFunctionRange(const std::string& Target);

  private:
    /// Inner class used by the ClangTool of the FunctionRangeExtractor
    class FunctionRangeActionFactory : public clang::tooling::FrontendActionFactory {
    private:
      FunctionRangeRes Results;
      FunctionRangeRes *UserProvidedResults;
      std::vector<std::string> Targets;
    public:
      FunctionRangeActionFactory();

      /// Retrieve the current target functions
      const std::vector<std::string>& getTargets() const;

      FunctionRangeActionFactory& clearTargets();
      FunctionRangeActionFactory& useTarget(const std::string& Target);
      FunctionRangeActionFactory& useTargets(const std::vector<std::string>& Targets);
      FunctionRangeActionFactory& addTarget(std::string& Target);
      FunctionRangeActionFactory& addTargets(const std::vector<std::string>& Targets);

      /// Provide a container to fill with the results. It should be used before
      /// running a ClangTool that makes use of this factory.
      FunctionRangeActionFactory& useResults(FunctionRangeRes& ResContainer);

      /// Retrieve the Results container used. In general the results are
      /// relevant to the latest invocation of the ClangTool that produces
      /// them. i.e calling this method before running the tool will just
      // return an empty result
      const FunctionRangeRes& getResults() const;

      /// ClangTool calls this on each run() invocation
      std::unique_ptr<clang::FrontendAction> create() override;
    };
};

} // namespace kerma

#endif