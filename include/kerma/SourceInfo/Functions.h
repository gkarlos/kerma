#ifndef KERMA_SOURCEINFO_FUNCTIONS_H
#define KERMA_SOURCEINFO_FUNCTIONS_H

#include "kerma/SourceInfo/SourceInfo.h"

#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>

namespace kerma {

class FunctionRangeAction : public clang::ASTFrontendAction {
private:
  std::unordered_map<std::string, SourceRange> &Res;
  std::vector<std::string> &Targets;
public:
  FunctionRangeAction( std::unordered_map<std::string, SourceRange> &Res,
                        std::vector<std::string>& Targets)
  : Res(Res), Targets(Targets){}

  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer( clang::CompilerInstance &CI,
                                                                  llvm::StringRef file) override;
};

class FunctionRangeActionFactory : public clang::tooling::FrontendActionFactory {
private:
  std::unordered_map<std::string, SourceRange> Results;
  std::vector<std::string> Targets;
public:
  FunctionRangeActionFactory() {}
  std::unordered_map<std::string, SourceRange> getResults() { return Results; }
  std::unique_ptr<clang::FrontendAction> create() override {
    Results.clear();;
    return std::make_unique<FunctionRangeAction>(Results, Targets);
  }
};

}



#endif // KERMA_SOURCEINFO_FUNCTIONS_H