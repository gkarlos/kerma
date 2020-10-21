#ifndef KERMA_ANALYSIS_NAMES_H
#define KERMA_ANALYSIS_NAMES_H

#include "llvm/IR/Value.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>
#include <llvm/PassAnalysisSupport.h>

#include <map>

namespace kerma {

/// This class is just a wrapper for LLVM metadata functionality
/// It merely makes it easy to access names for e.g global variables
/// or allocas and avoids repeatedly walking the metadata by caching.
class Namer {
protected:
  Namer()=delete;
public:
  static const std::string& None;
  static const std::string& GetNameForGlobal(llvm::GlobalVariable *GV, bool Gen=true);
  static const std::string& GetNameForAlloca(llvm::AllocaInst *Alloca, bool Gen=true);
  static const std::string& GetNameForArg(llvm::Argument *Arg, bool Gen=true);
};

} // namespace kerma

#endif // KERMA_ANALYSIS_NAMES_H