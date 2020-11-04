#include "kerma/SourceInfo/Functions.h"
#include "kerma/SourceInfo/SourceRange.h"
#include "kerma/Support/CXXExtras.h"
#include "kerma/SourceInfo/Util.h"
#include "clang/Basic/SourceLocation.h"

#include <clang/AST/Decl.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>


namespace kerma {

using namespace clang;

class FunctionRangeVisitor : public clang::RecursiveASTVisitor<FunctionRangeVisitor> {
public:
  FunctionRangeVisitor( clang::CompilerInstance &CI,
                        std::unordered_map<std::string, SourceRange> &Res,
                        std::vector<std::string> &Targets )
  : CI(CI), Context(CI.getASTContext()), SourceManager(CI.getSourceManager()),
    Res(Res), Targets(Targets) {}

  bool VisitFunctionDecl(clang::FunctionDecl* F) {
    // skip includes
    if ( !SourceManager.isInMainFile(F->getBeginLoc())) return true;
    if ( !F->isThisDeclarationADefinition()) return true;
    // if we have targets and this decl is not one bail
    if ( !Targets.empty() && !inVector(F->getName(), Targets))
      return true;

    // if ( F->getIdentifier())
    //   llvm::errs() << F->getIdentifier()->getName() << '\n';

    SourceLoc Begin(SourceManager.getPresumedLineNumber(F->getBeginLoc()),
                    SourceManager.getPresumedColumnNumber(F->getBeginLoc()));
    SourceLoc End(SourceManager.getPresumedLineNumber(F->getEndLoc()),
                  SourceManager.getPresumedColumnNumber(F->getEndLoc()));
    Res[F->getNameAsString()] = SourceRange(Begin, End);

    return true;
  }

private:
  clang::CompilerInstance &CI;
  clang::ASTContext &Context;
  clang::SourceManager &SourceManager;
  std::unordered_map<std::string, SourceRange>& Res;
  std::vector<std::string> Targets;
};



class FunctionRangeConsumer : public clang::ASTConsumer {
public:
  FunctionRangeConsumer( clang::CompilerInstance &CI,
                         std::unordered_map<std::string, SourceRange> &Res,
                         std::vector<std::string>& Targets)
  : Visitor(CI, Res, Targets), CI(CI) {}

  void HandleTranslationUnit(clang::ASTContext &Context) {
    CI.getDiagnosticOpts().ShowCarets = true;
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  FunctionRangeVisitor Visitor;
  clang::CompilerInstance &CI;
};



std::unique_ptr<clang::ASTConsumer>
FunctionRangeAction::CreateASTConsumer( clang::CompilerInstance &CI,
                                        llvm::StringRef file) {
  return std::make_unique<FunctionRangeConsumer>(CI, Res, Targets);
}

} // namespace kerma


