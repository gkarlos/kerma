#ifndef KERMA_SOURCEINFO_SOURCE_INFO_ACTION_FACTORY_H
#define KERMA_SOURCEINFO_SOURCE_INFO_ACTION_FACTORY_H

#include "kerma/SourceInfo/SourceInfo.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>

#include <iterator>
#include <memory>
#include <set>
#include <vector>

namespace kerma {

/// Objects of this class are passed to ClangTool's run().
/// Its job is to create a SourceInfoAction for the tool's
/// invocation, which will subsequently created the AST
/// Consumer and Visitor.
/// The SourceInfoFactory owns a SourceInfo object that
/// is being reused through Tool->run() invocations
/// or can be assigned (and hold a pointer) to an external
/// SourceInfo object through useSourceInfo().
/// The SourceInfo after a tool invocation can be accessed
/// with getSourceInfo();
/// In general, consecutive getSourceInfo() calls, are not
/// meant to return identical results
/// Care should be taken when holding a reference to an old
/// SourceInfo. Subsequent Tool.run(factory) invocations with
/// with the same factory may invalidate the old data.
/// If in doubt just take a copy of the SourceInfo.
class SourceInfoActionFactory : public clang::tooling::FrontendActionFactory {
private:
  SourceInfo DefaultSI;
  SourceInfo *ProvidedSI;
  /// Points to the currently "active" SI
  SourceInfo *SI;
  std::set<std::string> Targets;

public:
  SourceInfoActionFactory(SourceInfo &SI,
                          const std::vector<std::string> &Targets = {});
  SourceInfoActionFactory(const std::vector<std::string> &Targets = {});
  void addTarget(const std::string &Target);
  void addTargets(const std::vector<std::string> &Targets);
  void addTargets(const std::set<std::string> &Targets);
  void removeTarget(const std::string &Target);
  void clearTargets();
  void useSourceInfo(SourceInfo &SI);
  void useDefaultSourceInfo();
  bool isUsingDefaultSourceInfo();
  const SourceInfo &getSourceInfo();

  virtual std::unique_ptr<clang::FrontendAction> create() override;
};

class SourceInfoAction : public clang::ASTFrontendAction {
private:
  SourceInfo &SI;
  std::set<std::string> &Targets;

public:
  SourceInfoAction(SourceInfo &SI, std::set<std::string> &Targets);
  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef file) override;
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCE_INFO_ACTION_FACTORY_H