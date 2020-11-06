#include "kerma/SourceInfo/SourceInfoBuilder.h"
// #include "kerma/SourceInfo/Functions.h"
#include "kerma/SourceInfo/SourceInfoAction.h"
#include "kerma/Support/FileSystem.h"
// #include "clang/Frontend/CompilerInstance.h"

#include <boost/filesystem.hpp>
#include <memory>


// #include <clang/Frontend/TextDiagnosticPrinter.h>

namespace fs = boost::filesystem;
namespace kerma {

using namespace clang;
using namespace clang::tooling;

SourceInfoBuilder::SourceInfoBuilder( const std::string &SourcePath,
                                      clang::tooling::CompilationDatabase *DB)
{
  if ( !fs::exists(SourcePath))
    throw SourcePath + " does not exist";
  if ( fs::is_directory(SourcePath))
    throw SourcePath + " is a directory";

  if ( !DB) {
    std::string Err = "Could not read compile_commands.json";
    auto db = CompilationDatabase::autoDetectFromSource(SourcePath,Err);
    if ( !db) throw Err;
    CompileDB = db.release();
  }

  auto RealSourcePath = get_realpath(SourcePath);
  if ( RealSourcePath.empty())
    throw std::runtime_error(std::string("Error getting real path for ") + SourcePath);
  /// The requested file must have an entry in the Compilation Database
  if ( auto DBFiles = CompileDB->getAllFiles();
       std::find(DBFiles.begin(), DBFiles.end(), RealSourcePath) == DBFiles.end())
    throw std::runtime_error(RealSourcePath + " not found in CompilationDatabase");

  // this->SourcePath = RealSourcePath;
  SourcePaths.push_back(RealSourcePath);
  Tool = std::make_unique<clang::tooling::ClangTool>(*CompileDB, SourcePaths);
  // FunctionRanges = std::make_unique<FunctionRangeActionFactory>();
}


SourceInfo SourceInfoBuilder::getSourceInfo() {
  Tool->run(&ActionFactory);
  return ActionFactory.getSourceInfo();
}

} // namespace kerma
