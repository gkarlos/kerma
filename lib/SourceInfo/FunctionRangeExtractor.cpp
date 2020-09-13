#include "kerma/Compile/CompilationAdjuster.h"
#include "kerma/Compile/DiagnosticConsumers.h"
#include "kerma/SourceInfo/FunctionRangeExtractor.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"
#include "kerma/SourceInfo/Util.h"
#include "kerma/Support/CXXExtras.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace llvm;
using namespace clang;

namespace kerma {

namespace {
  /// This visitor will record the source range of every function in a file
  /// Clang treats both defs and decls as decls. Basically def = decl + body.
  /// To distringuish the two when we encounter a decl without a body we insert
  /// record its start location but the end location in Unknown.
  /// For function defs both locations are recorded.
  class FunctionRangeVisitor : public RecursiveASTVisitor<FunctionRangeVisitor> {
  private:
    ASTContext &Context;
    SourceManager &SourceManager;
    FunctionRangeExtractor::Result &Res;
    std::vector<std::string> &Targets;
  public:
    explicit FunctionRangeVisitor(CompilerInstance *CI, FunctionRangeExtractor::Result &Res, std::vector<std::string> &Targets)
    : Context(CI->getASTContext()), SourceManager(CI->getSourceManager()), Res(Res), Targets(Targets) {}

    bool VisitFunctionDecl(clang::FunctionDecl* F) {
      FullSourceLoc FullLocation = Context.getFullLoc(F->getLocation());

      if ( FullLocation.isValid() && !SourceManager.isInSystemHeader(FullLocation)
                                  && SourceManager.isInMainFile(F->getLocation())) {
        if ( !Targets.empty() && !inVector(F->getName(), Targets))
          return true;

        SourceRange range = kerma::readClangSrcRange(F->getSourceRange(), SourceManager);

        auto val = Res.find(F->getName().str());

        if ( val != Res.end())
          Res[F->getName().str()].push_back(range);
        else
          Res.insert({F->getName().str(), {range}});
      }
      return true;
    }
  };

  class FunctionRangeConsumer : public ASTConsumer {
  private:
    FunctionRangeVisitor Visitor;
    CompilerInstance *CI;

  public:
    explicit FunctionRangeConsumer(CompilerInstance* CI, FunctionRangeExtractor::Result &Res, std::vector<std::string>& Targets)
    : Visitor(CI, Res, Targets), CI(CI) {}

    virtual void HandleTranslationUnit(ASTContext &Context) override {
      // Make the CompileInstance not print a summary of the errors
      CI->getDiagnosticOpts().ShowCarets = false;
      Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
  };


  class FunctionRangeAction : public ASTFrontendAction {
  private:
    FunctionRangeExtractor::Result& Res;
    std::vector<std::string>& Targets;

  public:
    FunctionRangeAction(FunctionRangeExtractor::Result &Res, std::vector<std::string>& Targets): Res(Res), Targets(Targets) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer( CompilerInstance &CI, StringRef file) override {
      return std::make_unique<FunctionRangeConsumer>(&CI, Res, Targets);
    }
  };
} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

FunctionRangeExtractor::FunctionRangeActionFactory::FunctionRangeActionFactory()
: UserProvidedResults(nullptr)
{}

const std::vector<std::string>&
FunctionRangeExtractor::FunctionRangeActionFactory::getTargets() const {
  return Targets;
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::clearTargets() {
  Targets.clear();
  return *this;
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::useTarget(const std::string &Target) {
  Targets.clear();
  Targets.push_back(Target.c_str());
  return *this;
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::useTargets(const std::vector<std::string>& Targets) {
  this->Targets.clear();
  return addTargets(Targets);
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::addTarget(std::string &Target) {
  Targets.push_back(Target);
  return *this;
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::addTargets(const std::vector<std::string>& Targets) {
  for ( auto& T : Targets)
    this->Targets.push_back(T);
  return *this;
}

FunctionRangeExtractor::FunctionRangeActionFactory&
FunctionRangeExtractor::FunctionRangeActionFactory::useResults(Result& Results) {
  UserProvidedResults = &Results;
  return *this;
}

const FunctionRangeExtractor::Result&
FunctionRangeExtractor::FunctionRangeActionFactory::getResults() const {
  if ( UserProvidedResults)
    return *UserProvidedResults;
  return Results;
}

std::unique_ptr<FrontendAction>
FunctionRangeExtractor::FunctionRangeActionFactory::create() {
  if ( UserProvidedResults)
    return std::make_unique<FunctionRangeAction>(*UserProvidedResults, Targets);

  Results.clear();
  return std::make_unique<FunctionRangeAction>(Results, Targets);
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

FunctionRangeExtractor::FunctionRangeExtractor( std::string SourcePath,
                                                clang::tooling::CompilationDatabase *DB)
: CompileDB(DB) {
  SourcePaths.push_back(SourcePath);
  Tool = std::make_unique<tooling::ClangTool>(*CompileDB, SourcePaths);
  CompilationAdjuster::appendClangIncludes(*Tool);
  ActionFactory = std::make_unique<FunctionRangeActionFactory>();
}

unsigned int FunctionRangeExtractor::runTool() const {
  ErrorCountConsumer DiagnosticsConsumer;
  Tool->setDiagnosticConsumer(&DiagnosticsConsumer);
  Tool->run(ActionFactory.get());
  return DiagnosticsConsumer.getNumErrors();
}

void FunctionRangeExtractor::getFunctionRanges(Result& res) {
  ActionFactory->useResults(res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

void FunctionRangeExtractor::getFunctionRanges(std::vector<std::string>& Targets, Result& Res) {
  ActionFactory->useTargets(Targets);
  ActionFactory->useResults(Res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

void FunctionRangeExtractor::getFunctionRange(const std::string& Target, Result &Res) {
  ActionFactory->useTarget(Target);
  ActionFactory->useResults(Res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

const FunctionRangeExtractor::Result& FunctionRangeExtractor::getFunctionRanges() const {
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
  return ActionFactory->getResults();
}

const  FunctionRangeExtractor::Result& FunctionRangeExtractor::getFunctionRanges(const std::vector<std::string> &Targets) {
  ActionFactory->useTargets(Targets);
  return getFunctionRanges();
}

const  FunctionRangeExtractor::Result& FunctionRangeExtractor::getFunctionRange(const std::string& Target) {
  ActionFactory->useTarget(Target);
  return getFunctionRanges();
}

} // namespace kerma
