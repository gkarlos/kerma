#include "kerma/Support/SourceInfo.h"

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>


static std::string readFileContents(const char *file) {
  std::ifstream ifs(file);
  if (!ifs.good())
    throw std::runtime_error(std::string("Cannot read file ") + file);
  return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

//https://opensource.apple.com/source/clang/clang-425.0.24/src/tools/clang/docs/RAVFrontendAction.html

namespace kerma { 

using namespace llvm;
using namespace clang;

class FunctionRangeVisitor : public RecursiveASTVisitor<FunctionRangeVisitor> {
public:
  explicit FunctionRangeVisitor(CompilerInstance *CI) 
  : Context(&(CI->getASTContext())) 
  {}
private:
  ASTContext *Context;
};

class FunctionRangeConsumer : public ASTConsumer {
public:
  explicit FunctionRangeConsumer(CompilerInstance *CI)
  : Visitor(std::make_unique<FunctionRangeVisitor>(CI))
  {}
private:
  std::unique_ptr<FunctionRangeVisitor> Visitor;
};

class FunctionRangeAction : public ASTFrontendAction {
public:
  virtual std::unique_ptr<ASTConsumer> 
  CreateASTConsumer( CompilerInstance &CI, StringRef file) override {
    errs() << "Creating ASTConsumer for FILE:\n\n" << file.str() << "\nDONE";
    return std::unique_ptr<ASTConsumer>(new FunctionRangeConsumer(&CI));
  }
};

//https://github.com/anirudhSK/libclang-samples/blob/master/src/util.cc

std::string SourceInfo::COMPILEDB_READ_ERR_MSG("Failed to read compilation database file");

SourceInfo::SourceInfo(llvm::Module *M, tooling::CompilationDatabase *compileDB) 
  : M(M) 
{
  init();
}

void 
SourceInfo::init() {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  for ( auto &namedMD : this->M->getNamedMDList()) {
    if (  auto *compileUnitMD = dyn_cast<DICompileUnit>(namedMD.getOperand(0)) ) {
      this->CompileUnitMD = compileUnitMD;
      break;
    }
  }

  std::string fullPath(this->getDirectory() + "/" + this->getFilename());
  this->Src = readFileContents(fullPath.c_str());
}

std::string
SourceInfo::getFilename() {
  return this->CompileUnitMD->getFilename().str();
}

std::string
SourceInfo::getDirectory() {
  return this->CompileUnitMD->getDirectory().str();
}

std::string
SourceInfo::getFullPath() {
  return this->getDirectory() + "/" + this->getFilename();
}

std::string
SourceInfo::getSource() { 
  return this->Src;
}

SourceRange 
SourceInfo::getFunctionRange(const char *) {
  SmallVector<std::string, 4> files;
  files.push_back(this->getFullPath());
  ArrayRef<std::string> filesArr(files);

  tooling::ClangTool tool(*(this->CompileDB), filesArr);

  tool.run(tooling::newFrontendActionFactory<FunctionRangeAction>().get());
  return SourceRange();
}

// StringRef getFile() 
// {

// }



// SourceRange
// getFunctionDeclRange(Function *F) 
// {

// }

// SourceRange
// getInstructionStmtRange(Instruction *I) 
// {

// }

// SourceRange
// getStoreLhsRange(StoreInst *SI) 
// {

// }

// SourceRange
// getStoreRhsRange(StoreInst *SI) 
// {

// }


// unsigned int
// getNumStatementsAtline(unsigned int lineno) 
// {

// }



} // namespace kerma