#ifndef KERMA_SOURCEINFO_STMT_RANGE_EXTRACTOR_H
#define KERMA_SOURCEINFO_STMT_RANGE_EXTRACTOR_H

#include "kerma/SourceInfo/SourceRange.h"

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include <memory>
#include <unordered_map>

namespace kerma {

using StmtRangeRes = std::unordered_map< std::string, std::vector<kerma::SourceRange>>;

/// This class provides an API for extracting source code ranges
/// for statements in functions
class StmtRangeExtractor {
public:
  using Result = StmtRangeRes;

private:
  class StmtRangeActionFactory; // forward declaration
  std::vector<std::string> SourcePaths;
  clang::tooling::CompilationDatabase *CompileDB;
  std::unique_ptr<clang::tooling::ClangTool> Tool;
  std::unique_ptr<StmtRangeActionFactory> ActionFactory;

private:
  unsigned int runTool() const;

public:
  StmtRangeExtractor(std::string SourcePath, clang::tooling::CompilationDatabase *DB);
  void getStmtRanges(StmtRangeRes& Res);
  void getStmtRanges(std::vector<std::string> & Targets, StmtRangeRes &Res);
  void getStmtRanges(const std::string& Target, StmtRangeRes &Res);
  const StmtRangeExtractor::Result& getStmtRanges(const std::vector<std::string> &Targets={});
  const StmtRangeExtractor::Result& getStmtRanges(const std::string& Target);

private:
  class StmtRangeActionFactory : public clang::tooling::FrontendActionFactory {
  private:
    StmtRangeRes Results;
    StmtRangeRes *UserProvidedResults;
    std::vector<std::string> Targets;
  public:
    StmtRangeActionFactory();
    StmtRangeActionFactory& clearTargets();
    StmtRangeActionFactory& useTarget(const std::string& Target);
    StmtRangeActionFactory& useTargets(const std::vector<std::string&> Targets);
    StmtRangeActionFactory& addTarget(std::string& Target);
    StmtRangeActionFactory& addTargets(const std::vector<std::string>& Targets);
    StmtRangeActionFactory& useResults(StmtRangeRes& ResContainer);
    StmtRangeActionFactory& removeResults();
    const std::vector<std::string>& getTargets() const;
    const StmtRangeRes& getResults() const;
    /// ClangTool calls this on each run() invocation
    std::unique_ptr<clang::FrontendAction> create() override;
  };
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_STMT_RANGE_EXTRACTOR_H
