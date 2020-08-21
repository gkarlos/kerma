#include "llvm/Support/Error.h"
#define DEBUG_TYPE "MaterializeGridPass"

#include "kerma/Pass/MaterializeGrid.h"
#include "kerma/Pass/DetectKernels.h"
#include "kerma/Support/Demangle.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

using namespace llvm;

#ifdef KERMA_OPT_PLUGIN

#include "llvm/Support/CommandLine.h"

/// Set up some cl args for Opt
cl::OptionCategory MGOptCategory("Kerma Materialize-Grid Options (--kerma-mg)");
cl::opt<std::string> MGGrid("mg-grid", cl::Required,
                            cl::desc("Grid dimensions"), 
                            cl::value_desc("x,y,z"), cl::cat(MGOptCategory));
cl::opt<std::string> MGBlock("mg-block", cl::Required,
                            cl::desc("Block dimensions"), 
                            cl::value_desc("x,y,z"), cl::cat(MGOptCategory));
cl::opt<std::string> MGTarget("mg-target", cl::Optional,
                            cl::desc("Target kernel function"),
                            cl::value_desc("kernel_name"), cl::cat(MGOptCategory), cl::init(""));

kerma::Dim parseDimOrExit(const std::string& dim, const char *ErrorMsg="") {
  std::stringstream ss(dim);
  std::stringstream errss;
  std::vector<unsigned int> vals;
  try {
    while(ss.good()) {
      std::string substr;
      getline(ss, substr, ',');
      vals.push_back(std::stoul(substr, 0, 10));
    }
    if ( vals.size() != 3)
      throw "error";
  } catch ( ... ) {
    errss << "--kerma-mg: " << ErrorMsg;
    llvm::report_fatal_error(errss.str());
  }
  kerma::Dim res(vals[0], vals[1], vals[2]);
  return res;
}

#endif // KERMA_OPT_PLUGIN

static RegisterPass<kerma::MaterializeGridPass> RegisterLoopInfoTestPass(
        /* pass arg  */   "kerma-mg", 
        /* pass name */   "Materialize Grid and Block values in kernel functions", 
        /* modifies CFG */ false, 
        /* analysis pass*/ true);


namespace kerma {

char MaterializeGridPass::ID = 3;

#ifdef KERMA_OPT_PLUGIN

MaterializeGridPass::MaterializeGridPass() : MaterializeGridPass(parseDimOrExit( MGGrid.getValue(), "Invalid Grid"),
                                                                 parseDimOrExit( MGBlock.getValue(), "Invalid Block"))
{}

#else

MaterializeGridPass::MaterializeGridPass() : MaterializeGridPass(Dim::None, Dim::None)
{}

#endif

MaterializeGridPass::MaterializeGridPass(const Dim& Grid, const Dim& Block)
: Grid(Grid), Block(Block), TargetKernelFun(nullptr), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

MaterializeGridPass::MaterializeGridPass(const Dim& Grid, const Dim& Block, llvm::Function &F)
: Grid(Grid), Block(Block), TargetKernelFun(&F), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

void MaterializeGridPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  // llvm::errs() << "Called" << "\n";
  // AU.addRequired<DetectKernelsPass>();
  // AU.setPreservesAll();
}

void MaterializeGridPass::print(llvm::raw_ostream &O, const llvm::Module *M) const {

}

bool MaterializeGridPass::doInitialization(llvm::Module &M) {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "Materialize Grid to: " << this->Grid << '\n';
  llvm::errs() << "Materialize Block to: " << this->Block << '\n';
  llvm::errs() << "Target kernel: " << (!MGTarget.getValue().empty()? MGTarget.getValue() : "all") << '\n';
#endif

  if ( !hasWork()) {
    // TODO: pass may have been called from opt
    // Check if there are flags set
    llvm::errs() << "No work!\n";
  }

  llvm::ExitOnError("asdasdasd");
  // this->Kernels = getAnalysis<DetectKernelsPass>().getKernels();
  DetectKernelsPass DetectKernels;
  DetectKernels.runOnModule(M);
  this->Kernels = DetectKernels.getKernels();

  return false;
}

bool MaterializeGridPass::doFinalization(llvm::Module &M) {
  return false;
}

bool MaterializeGridPass::isTargeted() {
  return this->TargetKernelFun != nullptr && this->TargetKernelName != nullptr;
}

bool MaterializeGridPass::isKernel(llvm::Function &F) {
  return std::find(this->Kernels.begin(), this->Kernels.end(), &F) != this->Kernels.end();
}


const std::string gridDim("__cuda_builtin_gridDim_t");
const std::string blockDim("__cuda_builtin_blockDim_t");
const std::string x("__fetch_builtin_x()");
const std::string y("__fetch_builtin_y()");
const std::string z("__fetch_builtin_z()");

bool isBlockDimCall(llvm::Function &F) {
  return demangleFn(F).find(blockDim) != std::string::npos;
}

bool isGridDimCall(llvm::Function &F) {
  return demangleFn(F).find(gridDim) != std::string::npos;
}

bool isDimXCall(llvm::Function &F) {
  return demangleFn(F).find(x) != std::string::npos;
}

bool isDimYCall(llvm::Function &F) {
  return demangleFn(F).find(y) != std::string::npos;
}

bool isDimZCall(llvm::Function &F) {
  return demangleFn(F).find(z) != std::string::npos;
}

bool MaterializeGridPass::hasWork() const {
  return this->Grid == Dim::None && this->Block == Dim::None;
}

ConstantInt *createUnsignedInt(LLVMContext &context, unsigned int value) {
  auto *ty = IntegerType::get(context, 32);
  return ConstantInt::get(ty, value, false);
}

bool MaterializeGridPass::analyzeKernel(llvm::Function &F) const {
  bool changed = false;

  for ( auto &BB : F) {
    for ( auto &I : BB) {
      if ( auto *CI = dyn_cast<CallInst>(&I)) {
        
        auto *callee = CI->getCalledFunction();
        
        unsigned int NumUses = I.getNumUses();

        if ( isGridDimCall(*callee)) {
          
          if ( isDimXCall(*callee)) {
          } else if ( isDimYCall(*callee)) {
          } else if ( isDimZCall(*callee)) {
          } else {
            // should not happen
          }
          changed = changed || NumUses;

        } else if ( isBlockDimCall(*callee)) {

          if ( isDimXCall(*callee)) {
          } else if ( isDimYCall(*callee)) {
          } else if ( isDimZCall(*callee)) {
          } else {
            // should not happen
          }
          changed = changed || NumUses;

        }
      }
    }
  }
  return changed;
}

bool MaterializeGridPass::runOnFunction(llvm::Function &F) {

  if ( F.isDeclaration() || F.isIntrinsic())
    return false;
  if ( !this->isKernel(F))
    return false;
    
  if ( this->isTargeted()) {
    if ( this->TargetKernelFun == &F || this->TargetKernelName == demangleFnWithoutArgs(F)) {}
      //TODO also check if name passed as arg in opt
      // return analyzeKernel(F);
  } else {
    // return analyzeKernel(F);
  }
  return false;
}

}