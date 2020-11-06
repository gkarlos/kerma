#ifndef KERMA_SOURCEINFO_SOURCE_INFO_BUILDER_H
#define KERMA_SOURCEINFO_SOURCE_INFO_BUILDER_H

#include "kerma/SourceInfo/SourceInfoAction.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <clang/AST/ASTContext.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace kerma {

/// A facade, hiding  all the underlying mechanisms
/// required to create a SourceInfo object.
/// A SourceInfoBuilder is associated with a single
/// File/CompileDB
/// The API is a single function getSourceInfo()
/// which does all the work and returns a SourceInfo
/// object by value. The function is meant to be
/// called infrequently
class SourceInfoBuilder {
public:
  SourceInfoBuilder(const std::string &SourcePath,
                    clang::tooling::CompilationDatabase *DB = nullptr);

private:
  SourceInfo SI;
  clang::tooling::CompilationDatabase *CompileDB;
  std::vector<std::string> SourcePaths;
  std::unique_ptr<clang::tooling::ClangTool> Tool;
  SourceInfoActionFactory ActionFactory;

public:
  SourceInfo getSourceInfo();
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCE_INFO_BUILDER_H