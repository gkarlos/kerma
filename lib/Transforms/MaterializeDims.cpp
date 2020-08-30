#define DEBUG_TYPE "MaterializeDimsPass"

#include "kerma/Transforms/MaterializeDims.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/Support/Demangle.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace llvm;

#ifdef KERMA_OPT_PLUGIN

#include "llvm/Support/CommandLine.h"

/// Set up some cl args for Opt
cl::OptionCategory MDOptionCategory("Kerma Materialize-Dim Options (--kerma-md)");
cl::opt<std::string> MDGrid("md-grid", cl::desc("Grid dimensions. Default=1,1,1"), 
                            cl::value_desc("x,y,z"), cl::init("1,1,1"), cl::cat(MDOptionCategory));
cl::opt<std::string> MDBlock("md-block", cl::desc("Block dimensions. Default=1,1,1"), 
                            cl::value_desc("x,y,z"), cl::init("1,1,1"), cl::cat(MDOptionCategory));
cl::opt<std::string> MDTarget("md-target", cl::Optional,
                            cl::desc("Target kernel function"),
                            cl::value_desc("kernel_name"), cl::cat(MDOptionCategory), cl::init(""));

/// Parse a string of the form x,y,z into a Dim(x,y,z).If the parsing fails 
/// an llvm fatal_error is issued and the program exits with an appropriate 
/// message. This function is not meant to be used by library code but rather 
/// only for testing purposes in Opt plugins.
kerma::Dim parseDimOrExit(const std::string& DimStr, const char *ErrorMsg="") {
  std::stringstream ss(DimStr);
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
    errss << "--kerma-md: " << ErrorMsg;
    llvm::report_fatal_error(errss.str());
  }
  kerma::Dim res(vals[0], vals[1], vals[2]);
  return res;
}

#endif // KERMA_OPT_PLUGIN

static RegisterPass<kerma::MaterializeDimsPass> RegisterLoopInfoTestPass(
        /* pass arg  */   "kerma-md", 
        /* pass name */   "Materialize Grid and Block dimensions in kernels", 
        /* modifies CFG */ false, 
        /* analysis pass*/ true);


namespace kerma {

char MaterializeDimsPass::ID = 3;

#ifdef KERMA_OPT_PLUGIN
MaterializeDimsPass::MaterializeDimsPass() : MaterializeDimsPass(parseDimOrExit( MDGrid.getValue(), "Invalid Grid"),
                                                                 parseDimOrExit( MDBlock.getValue(), "Invalid Block"))
{}
#else
MaterializeDimsPass::MaterializeDimsPass() : MaterializeDimsPass(Dim::Unit, Dim::Unit)
{}
#endif

MaterializeDimsPass::MaterializeDimsPass(const Dim& Grid, const Dim& Block)
: Grid(Grid), Block(Block), TargetKernelFun(nullptr), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

MaterializeDimsPass::MaterializeDimsPass(const Dim& Grid, const Dim& Block, llvm::Function &F)
: Grid(Grid), Block(Block), TargetKernelFun(&F), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

MaterializeDimsPass::MaterializeDimsPass(const Dim& Grid, const Dim& Block, const char *KernelName)
: Grid(Grid), Block(Block), TargetKernelFun(nullptr), TargetKernelName(KernelName), llvm::FunctionPass(ID)
{}

void MaterializeDimsPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {}


bool MaterializeDimsPass::doInitialization(llvm::Module &M) {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "Materialize Grid to: " << this->Grid << '\n';
  llvm::errs() << "Materialize Block to: " << this->Block << '\n';
  llvm::errs() << "Target kernel: " << (!MDTarget.getValue().empty()? MDTarget.getValue() : "all") << "\n\n";

  if ( !this->hasWork())
    llvm::errs() << "Nothing to do\n";
  if ( !MDTarget.empty())
    this->TargetKernelName = MDTarget.getValue().c_str();
#endif

  if ( this->hasWork()) {
    DetectKernelsPass DetectKernels;
    DetectKernels.runOnModule(M);
    DetectKernels.getKernels(this->Kernels);
  }

  return false;
}

bool MaterializeDimsPass::doFinalization(llvm::Module &M) {
  return false;
}

bool MaterializeDimsPass::hasTargetKernel() const {
  return this->TargetKernelFun || this->TargetKernelName;
}

bool MaterializeDimsPass::isKernel(llvm::Function &F) {
  return (std::find(this->Kernels.begin(), this->Kernels.end(), &F) != this->Kernels.end())
      || nvvm::isKernelFunction(F);
}

bool MaterializeDimsPass::hasWork() const {
  return this->Grid && this->Block;
}

bool isBlockDimBuiltin(llvm::Function &F) {
  return llvm::demangle(F.getName()).find(nvvm::BlockDim) != std::string::npos;
}

bool isGridDimBuiltin(llvm::Function &F) {
  return llvm::demangle(F.getName()).find(nvvm::GridDim) != std::string::npos;
}

ConstantInt *createUnsignedInt(LLVMContext &context, unsigned int value) {
  auto *ty = IntegerType::get(context, 32);
  return ConstantInt::get(ty, value, false);
}

bool MaterializeDimsPass::analyzeKernel(llvm::Function &F) const {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "--Analyzing: " << llvm::demangle(F.getName()) << '\n';
#endif

  unsigned int changes = 0;

  for ( auto &BB : F) {
    for ( auto &I : BB) {
      if ( auto *CI = dyn_cast<CallInst>(&I)) {
        
        auto *Callee = CI->getCalledFunction();
        auto DemangledCalleeName = llvm::demangle(Callee->getName());
        
        if ( !isBlockDimBuiltin(*Callee) && !isGridDimBuiltin(*Callee))
          continue;

#ifdef KERMA_OPT_PLUGIN
        llvm::errs() << (isGridDimBuiltin(*Callee)? "  -grid.dim" : "  -block.dim") 
                     << " call at line " << CI->getDebugLoc().getLine();
#endif

        if ( nvvm::GridDim.x == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Grid.x));
        }
        else if ( nvvm::GridDim.y == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Grid.y));
        }
        else if ( nvvm::GridDim.z == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Grid.z));
        }
        else if ( nvvm::BlockDim.x == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.x));
        }
        else if ( nvvm::BlockDim.y == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.y));
        }
        else if ( nvvm::BlockDim.z == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.z));
        }

#ifdef KERMA_OPT_PLUGIN
        llvm::errs() << " -> " << Callee->getNumUses() << " uses materialized\n";
#endif 

        changes += Callee->getNumUses();
      }
    }
  }

  return changes;
}

bool MaterializeDimsPass::runOnFunction(llvm::Function &F) {

  if ( F.isDeclaration() || F.isIntrinsic())
    return false;

  if ( !this->isKernel(F))
    return false;

  if ( !this->hasWork())
    return false;
    
  if ( this->hasTargetKernel() ) {
    if ( !(this->TargetKernelFun && this->TargetKernelFun->getName() == F.getName()) &&  
         !(this->TargetKernelName && this->TargetKernelName == demangleFnWithoutArgs(F))) 
    {
#ifdef KERMA_OPT_PLUGIN
    llvm::errs() << "Skipping: " << llvm::demangle(F.getName()) << '\n';
#endif
      return false;
    }  
  }
  
  return analyzeKernel(F);
}


std::unique_ptr<MaterializeDimsPass> createMaterializeDimsPass() {
  return std::make_unique<MaterializeDimsPass>();
}

std::unique_ptr<MaterializeDimsPass> 
createMaterializeDimsPass(const Dim& Grid, const Dim& Block) {
  return std::make_unique<MaterializeDimsPass>(Grid, Block);
}

std::unique_ptr<MaterializeDimsPass>
createMaterializeDimsPass(const Dim& Grid, const Dim& Block, llvm::Function &F) {
  return std::make_unique<MaterializeDimsPass>(Grid, Block, F);
}

std::unique_ptr<MaterializeDimsPass>
createMaterializeDimsPass(const Dim& Grid, const Dim& Block, const char *KernelName) {
  return std::make_unique<MaterializeDimsPass>(Grid, Block, KernelName);
}

} // namespace kerma