#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include <kerma/Support/LLVMFunctionShorthands.h>

namespace kerma
{

int getFnNumArgs(llvm::Function &fn)
{
  return fn.arg_size();
}

llvm::Function *
getNextFunctionDefinition(llvm::Function &f)
{
  llvm::Function *result = nullptr;

  llvm::Module *module = f.getParent();
  
  bool found = false;

  if ( module != nullptr) {
    for ( auto &F : *module) {
      
      if ( found) {
        result = &F;
        break;
      }

      if ( static_cast<llvm::Value*>(&f) == static_cast<llvm::Value*>(&F))
        found = true;
    }
  }

  return result;
}

}