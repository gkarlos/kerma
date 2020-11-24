#include "kerma/SourceInfo/SourceInfoAction.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/Util.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Attrs.inc> // Cannot include this without Attr.h
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Basic/Cuda.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <tuple>

namespace kerma {

using namespace clang;
using namespace llvm;

/// The main Clang AST visitor, recording all the required
/// source code information. It is passed a SourceInfo obj,
/// which it populates.
class SourceInfoVisitor : public clang::StmtVisitor<SourceInfoVisitor> {
public:
  SourceInfoVisitor(clang::CompilerInstance &CI, SourceInfo &SI,
                    std::set<std::string> Targets)
      : CI(CI), Context(CI.getASTContext()),
        SourceManager(CI.getSourceManager()), SI(SI) {}

  /// Record the ranges of kernels and device functions
  /// and visit their bodies.
  bool VisitFunctionDecl(FunctionDecl *F) {
    if (F->isThisDeclarationADefinition()) {
      auto Range = GetSourceRange(SourceManager, *F);
      if (F->hasAttr<clang::CUDAGlobalAttr>())
        SI.KernelRanges[F->getNameAsString()] = Range;
      else if (F->hasAttr<CUDADeviceAttr>())
        SI.DeviceFunctionRanges[F->getNameAsString()] = Range;
      else
        return true;
      return VisitStmt(F->getBody());
    }
    return true;
  }

  bool VisitExpr(Expr *E) {
    if (!E)
      return true;
    SI.Exprs.push_back(GetSourceRange(SourceManager, *E));
    return true;
  }

  bool VisitStmt(Stmt *S) {
    if (!S)
      return true;

    if (auto *Compound = dyn_cast<CompoundStmt>(S)) {
      for (auto *C : Compound->body())
        VisitStmt(C);
    } else if (auto *If = dyn_cast<IfStmt>(S)) {
      SourceRange Cond, Then, Else;
      if ( If->getCond()) Cond = GetSourceRange(SourceManager, *If->getCond());
      if ( If->getThen()) Then = GetSourceRange(SourceManager, *If->getThen());
      if ( If->getElse()) Else = GetSourceRange(SourceManager, *If->getElse());

      SI.IfStmts.push_back(std::make_tuple(Cond, Then, Else));

      // errs() << "(IF) cond: " << Cond;
      // errs() << "     then: " << Then;
      // errs() << "     else: " << Else;
      // errs() << "\n";
      VisitStmt(If->getThen());
      if (auto *Else = If->getElse())
        VisitStmt(Else);
    } else if (auto *For = dyn_cast<ForStmt>(S)) {
      SI.ForInits.push_back(GetForStmtInitRange(SourceManager, *For));
      SI.ForHeaders.push_back(GetForStmtHeaderRange(SourceManager, *For));
      if (For->getBody())
        VisitStmt(For->getBody());
    } else if (auto *DoWhile = dyn_cast<DoStmt>(S)) {
      SI.DoConditions.push_back(
          GetSourceRange(SourceManager, *DoWhile->getCond()));
      if (DoWhile->getBody())
        VisitStmt(DoWhile->getBody());
    } else if (auto *While = dyn_cast<WhileStmt>(S)) {
      SI.WhileConditions.push_back(
          GetSourceRange(SourceManager, *While->getCond()));
      if (While->getBody())
        VisitStmt(While->getBody());
    } else {
      SI.Stmts.push_back(GetSourceRange(SourceManager, *S));
    }
    return true;
  }

  bool VisitTranslationUnit(TranslationUnitDecl *TU) {
    for (auto *D : TU->decls())
      if (SourceManager.isInMainFile(D->getBeginLoc())) {
        if (auto *F = dyn_cast<FunctionDecl>(D))
          VisitFunctionDecl(F);
        // extern "C" ...
        if (auto *L = dyn_cast<LinkageSpecDecl>(D)) {
          if (auto *f = dyn_cast<FunctionDecl>(*L->decls_begin()))
            VisitFunctionDecl(f);
        }
      }
    return true;
  }

private:
  clang::SourceManager &SourceManager;
  clang::CompilerInstance &CI;
  clang::ASTContext &Context;
  SourceInfo &SI;
};

class SourceInfoConsumer : public ASTConsumer {
public:
  SourceInfoConsumer(CompilerInstance &CI, SourceInfo &SI,
                     std::set<std::string> &Targets)
      : Visitor(CI, SI, Targets) {
    SI.clear();
    CI.getDiagnosticOpts().ShowCarets = true;
  }
  void HandleTranslationUnit(ASTContext &Context) {
    Visitor.VisitTranslationUnit(Context.getTranslationUnitDecl());
  }

private:
  SourceInfoVisitor Visitor;
};

// SourceInfoAction
SourceInfoAction::SourceInfoAction(SourceInfo &SI,
                                   std::set<std::string> &Targets)
    : SI(SI), Targets(Targets) {}

std::unique_ptr<clang::ASTConsumer>
SourceInfoAction::CreateASTConsumer(clang::CompilerInstance &CI,
                                    llvm::StringRef file) {
  return std::make_unique<SourceInfoConsumer>(CI, SI, Targets);
}

// SourceInfoActionFactory
SourceInfoActionFactory::SourceInfoActionFactory(
    const std::vector<std::string> &Targets)
    : ProvidedSI(nullptr), SI(&DefaultSI) {
  addTargets(Targets);
}

SourceInfoActionFactory::SourceInfoActionFactory(
    SourceInfo &SI, const std::vector<std::string> &Targets)
    : ProvidedSI(&SI), SI(&SI) {
  addTargets(Targets);
}

void SourceInfoActionFactory::addTarget(const std::string &Target) {
  Targets.insert(Target);
}

void SourceInfoActionFactory::addTargets(
    const std::vector<std::string> &Targets) {
  this->Targets.insert(Targets.begin(), Targets.end());
}

void SourceInfoActionFactory::addTargets(const std::set<std::string> &Targets) {
  this->Targets.insert(Targets.begin(), Targets.end());
}

void SourceInfoActionFactory::removeTarget(const std::string &Target) {
  Targets.erase(Target);
}

void SourceInfoActionFactory::clearTargets() { Targets.clear(); }

void SourceInfoActionFactory::useSourceInfo(SourceInfo &SI) {
  ProvidedSI = &SI;
  this->SI = ProvidedSI;
}

void SourceInfoActionFactory::useDefaultSourceInfo() {
  ProvidedSI = nullptr;
  this->SI = &DefaultSI;
}

bool SourceInfoActionFactory::isUsingDefaultSourceInfo() {
  return SI == &DefaultSI;
}

const SourceInfo &SourceInfoActionFactory::getSourceInfo() { return *SI; }

std::unique_ptr<clang::FrontendAction> SourceInfoActionFactory::create() {
  return std::make_unique<SourceInfoAction>(*SI, Targets);
}

} // namespace kerma