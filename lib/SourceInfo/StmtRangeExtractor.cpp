#include "kerma/SourceInfo/StmtRangeExtractor.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <memory>


using namespace llvm;
using namespace clang;

namespace kerma {

namespace {

  class StmtRangeVisitor : public clang::RecursiveASTVisitor<StmtRangeVisitor> {
  private:
    ASTContext &Context;
    SourceManager &SM;
    StmtRangeRes &Res;
    std::vector<std::string> &Targets;

  public:
    explicit StmtRangeVisitor(CompilerInstance *CI, StmtRangeRes &Res, std::vector<std::string> &Targets)
    : Context(CI->getASTContext()), SM(CI->getSourceManager()), Res(Res), Targets(Targets)
    {}
    bool VisitFunctionDecl(clang::FunctionDecl *F) {
      // TODO implement me
      return false;
    }
  };

  class StmtRangeConsumer : public ASTConsumer {
  private:
    StmtRangeVisitor Visitor;
    CompilerInstance *CI;
  public:
    explicit StmtRangeConsumer(CompilerInstance* CI, StmtRangeRes& Res, std::vector<std::string>& Targets)
    : Visitor(CI, Res, Targets), CI(CI)
    {}
    virtual void HandleTranslationUnit(ASTContext &Context) override {
      Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
  };

  class StmtRangeAction : public ASTFrontendAction {
  private:
    StmtRangeExtractor::Result& Res;
    std::vector<std::string>& Targets;
  public:
    StmtRangeAction()=delete;
    StmtRangeAction(StmtRangeExtractor::Result &Res, std::vector<std::string> &Targets): Res(Res), Targets(Targets) {}
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef File) override {
      return std::make_unique<StmtRangeConsumer>(&CI, Res, Targets); 
    }
  };
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

StmtRangeExtractor::StmtRangeActionFactory::StmtRangeActionFactory() : UserProvidedResults(nullptr) {}

const std::vector<std::string> &
StmtRangeExtractor::StmtRangeActionFactory::getTargets() const { return Targets; }

StmtRangeExtractor::StmtRangeActionFactory &
StmtRangeExtractor::StmtRangeActionFactory::clearTargets() {
  Targets.clear();
  return *this;
}

}