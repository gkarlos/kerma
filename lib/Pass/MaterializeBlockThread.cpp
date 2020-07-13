#include "kerma/Pass/MaterializeBlockThread.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/PassSupport.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Demangle/Demangle.h"

using namespace llvm;

const std::string blockDim("__cuda_builtin_blockDim_t");
const std::string blockIdx("__cuda_builtin_blockIdx_t");
const std::string threadIdx("__cuda_builtin_threadIdx_t");
const std::string x("__fetch_builtin_x()");
const std::string y("__fetch_builtin_y()");
const std::string z("__fetch_builtin_z()");

bool isBlockDimCall(Function *F) {
  return demangle(F->getName().str()).find(blockDim) != std::string::npos;
}

bool isBlockIdxCall(Function *F) {
  return demangle(F->getName().str()).find(blockIdx) != std::string::npos;
}

bool isThreadIdxCall(Function *F) {
  return demangle(F->getName().str()).find(threadIdx) != std::string::npos;
}

ConstantInt *createInt(LLVMContext &context, int value) {
  auto *ty = IntegerType::get(context, 32);
  return ConstantInt::get(ty, value);
}

namespace kerma {

MaterializeBlockThreadPass::MaterializeBlockThreadPass() : llvm::FunctionPass(ID)
{}

bool
MaterializeBlockThreadPass::runOnFunction(llvm::Function &F)
{
  for ( auto &BB : F) {
    for ( auto &I : BB) {
      if ( auto *CI = dyn_cast<CallInst>(&I)) {
        auto *callee = CI->getCalledFunction();
        
        if ( isBlockDimCall(callee)) {
          I.replaceAllUsesWith(createInt(F.getContext(), 10));
        }

        else if ( isBlockIdxCall(callee)) {
          I.replaceAllUsesWith(createInt(F.getContext(), 20));
        }

        else if ( isThreadIdxCall(callee)) {
          I.replaceAllUsesWith(createInt(F.getContext(), 30));
        } 
      }
    }
  }
  return true;
}

} // namespace kerma end

char kerma::MaterializeBlockThreadPass::ID = 2;
static llvm::RegisterPass<kerma::MaterializeBlockThreadPass> X("kerma-mbt", "Replace block and thread indices with concrete values", false, false);