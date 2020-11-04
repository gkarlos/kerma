#ifndef KERMA_SOURCEINFO_SOURCE_INFO_BUILDER_H
#define KERMA_SOURCEINFO_SOURCE_INFO_BUILDER_H

#include "kerma/SourceInfo/Functions.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"
// #include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTContext.h"
// #include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
// #include <clang/Tooling/CompilationDatabase.h>
// #include <clang/Tooling/Tooling.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace kerma {

// class StmtInfoActionFactory : public clang::tooling::FrontendActionFactory {
// private:
//   std::vector<std::string> Targets;
// };

class SourceInfoBuilder {
public:
  SourceInfoBuilder( const std::string &SourcePath,
                     clang::tooling::CompilationDatabase *DB = nullptr);

private:
  SourceInfo SI;
  clang::tooling::CompilationDatabase *CompileDB;
  std::vector<std::string> SourcePaths;
  std::unique_ptr<clang::tooling::ClangTool> Tool;
  // std::unique_ptr<StmtInfoActionFactory> StmtInfos;
  FunctionRangeActionFactory FunctionRanges;

public:
  SourceInfo getSourceInfo();
  // SourceInfo getSourceInfo() {
  //   SourceInfo SI;
  //   Tool->run(FunctionRanges.get());
  //   SI.addFunctions(FunctionRanges->getResults());
  //   return SI;
  // }

};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCE_INFO_LOCATOR_H