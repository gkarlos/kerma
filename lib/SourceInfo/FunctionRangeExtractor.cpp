#include "kerma/Compile/CompilationAdjuster.h"
#include "kerma/Compile/DiagnosticConsumers.h"
#include "kerma/SourceInfo/FunctionRangeExtractor.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include "clang/AST/DeclGroup.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace llvm;
using namespace clang;

namespace kerma {

namespace {
  // Read a clang::SourceLocation string of the form 'line:col'
  // into a kerma::SourceLoc. i.e simply parse 'line' and 'col'
  // into integers and create the SourceLoc
  void parseLoc( const std::string& LocStr, SourceLoc& Loc) {
    std::string token;
    std::vector<std::string> Values;
    std::stringstream iss(LocStr);

    while ( std::getline(iss, token, ':'))
      Values.push_back(token);

    Loc.set( std::stoul(Values[0]), std::stoul(Values[1]));
  }

  // Turn a clang::SourceRange to a kerma::SourceRange
  //
  // FIXME: A range string has the form:
  //        <filename>:<line>:<col>[ <other>]
  //                  ^             ^space
  // At the moment we extract the substr between the first ':' and the
  // first space. If no space exists, then until the end of the string.
  // This may fail on malformed strings so a more robust way is needed.
  SourceRange parseRange(const clang::SourceRange &Range, SourceManager& SourceManager) {
    std::string BeginLocStr = Range.getBegin().printToString(SourceManager);
    std::string EndLocStr = Range.getEnd().printToString(SourceManager);

    BeginLocStr = BeginLocStr.substr(BeginLocStr.find(':') + 1,
                                     BeginLocStr.find(' ') - (BeginLocStr.find(':')? BeginLocStr.find(':') + 1 : 0));
    EndLocStr = EndLocStr.substr(EndLocStr.find(':') + 1,
                                 EndLocStr.find(' ') - (EndLocStr.find(':')? EndLocStr.find(':') + 1 : 0));

    SourceRange res;

    try {
      parseLoc(BeginLocStr, res.getStart());
      parseLoc(EndLocStr, res.getEnd());
    } catch (...) {
      return SourceRange::Unknown;
    }

    return res;
  }

  /// This visitor will record the source range of every function in a file
  /// Clang treats both defs and decls as decls. Basically def = decl + body.
  /// To distringuish the two when we encounter a decl without a body we insert
  /// record its start location but the end location in Unknown.
  /// For function defs both locations are recorded.
  class FunctionRangeVisitor : public RecursiveASTVisitor<FunctionRangeVisitor> {
  private:
    ASTContext &Context;
    SourceManager &SourceManager;
    FunctionRangeRes &Res;
    std::vector<std::string> &Targets;
  public:
    explicit FunctionRangeVisitor(CompilerInstance *CI, FunctionRangeRes &Res, std::vector<std::string> &Targets)
    : Context(CI->getASTContext()), SourceManager(CI->getSourceManager()), Res(Res), Targets(Targets)
    {}

    bool VisitFunctionDecl(clang::FunctionDecl* F) {
      FullSourceLoc FullLocation = Context.getFullLoc(F->getLocation());
      if ( FullLocation.isValid() && !SourceManager.isInSystemHeader(FullLocation)
                                  && SourceManager.isInMainFile(F->getLocation())) {
        SourceRange range = parseRange(F->getSourceRange(), SourceManager);

        auto val = Res.find(F->getName().str());

        if ( val != Res.end())
          Res[F->getName().str()].push_back(range);
        else
          Res.insert({F->getName().str(), {range}});
      }
      return true;
    }
  };

  /// This consumer just iterates over all top level
  /// declarations in a file
  class FunctionRangeConsumer : public ASTConsumer {
  private:
    FunctionRangeVisitor Visitor;
    CompilerInstance *CI;

  public:
    explicit FunctionRangeConsumer(CompilerInstance* CI, FunctionRangeRes& Res, std::vector<std::string>& Targets)
    : Visitor(CI, Res, Targets), CI(CI)
    {}

    // bool HandleTopLevelDecl(DeclGroupRef D) override {
    //   for ( auto& Decl : D)
    //     Visitor.TraverseDecl(Decl);
    //   return true;
    // }

    virtual void HandleTranslationUnit(ASTContext &Context) override {
      // Make the CompileInstance not print a summary of the errors
      CI->getDiagnosticOpts().ShowCarets = false;
      // we can use ASTContext to get the TranslationUnitDecl, which is
      // a single Decl that collectively represents the entire source file
      Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
  };


  class FunctionRangeAction : public ASTFrontendAction {
  private:
    FunctionRangeRes &Res;
    std::vector<std::string>& Targets;

  public:
    FunctionRangeAction(FunctionRangeRes &Res, std::vector<std::string>& Targets): Res(Res), Targets(Targets)
    {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer( CompilerInstance &CI, StringRef file) override {
      return std::make_unique<FunctionRangeConsumer>(&CI, Res, Targets);
    }
  };
} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

FunctionRangeExtractor::FunctionRangeActionFactory
::FunctionRangeActionFactory() : UserProvidedResults(nullptr)
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
  Targets.push_back(Target);
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
FunctionRangeExtractor::FunctionRangeActionFactory::useResults(FunctionRangeRes& ResContainer) {
  UserProvidedResults = &ResContainer;
  return *this;
}

const FunctionRangeRes&
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
  // std::cout << DiagConsumer.getNumErrors() << "/" << DiagConsumer.getNumWarnings() << '\n';
  return DiagnosticsConsumer.getNumErrors();
}

void FunctionRangeExtractor::getFunctionRanges(FunctionRangeRes& res) {
  ActionFactory->useResults(res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

void FunctionRangeExtractor::getFunctionRanges(std::vector<std::string>& Targets, FunctionRangeRes& Res) {
  ActionFactory->useTargets(Targets);
  ActionFactory->useResults(Res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

void FunctionRangeExtractor::getFunctionRange(const std::string& Target, FunctionRangeRes &Res) {
  ActionFactory->useTarget(Target);
  ActionFactory->useResults(Res);
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
}

const FunctionRangeRes& FunctionRangeExtractor::getFunctionRanges() const {
  if ( auto err = runTool())
    throw std::runtime_error(std::to_string(err) + " errors while processing " + *SourcePaths.begin());
  return ActionFactory->getResults();
}

const FunctionRangeRes& FunctionRangeExtractor::getFunctionRanges(const std::vector<std::string> &Targets) {
  ActionFactory->useTargets(Targets);
  return getFunctionRanges();
}

const FunctionRangeRes& FunctionRangeExtractor::getFunctionRange(const std::string& Target) {
  ActionFactory->useTarget(Target);
  return getFunctionRanges();
}

} // namespace kerma
