#define DEBUG_TYPE "MaterializeIdxPass"

#include "kerma/Transforms/MaterializeIdx.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Base/Index.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/Support/Demangle.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Demangle/Demangle.h>
#include <iostream>
#include <sstream>
#include <vector>


using namespace llvm;

#ifdef KERMA_OPT_PLUGIN

/// Set up some cl args for Opt
cl::OptionCategory MIOptionCategory("Kerma Materialize-Idx Options (--kerma-mi)");
cl::opt<std::string> MIBlock("mi-block", cl::desc("Block index. Default=0,0,0"), 
                            cl::value_desc("z,y,x"), cl::init("0,0,0"), cl::cat(MIOptionCategory));
cl::opt<std::string> MIThread("mi-thread", cl::desc("Thread index. Default=0,0,0"), 
                            cl::value_desc("z,y,x"), cl::init("0,0,0"), cl::cat(MIOptionCategory));
cl::opt<std::string> MITarget("mi-target", cl::Optional,
                            cl::desc("Target kernel function"),
                            cl::value_desc("kernel_name"), cl::cat(MIOptionCategory), cl::init(""));

namespace {
  void llvmExit(const char *ErrorMsg="") {
    std::stringstream errss;
    errss << "--kerma-mi: " << ErrorMsg;
    llvm::report_fatal_error(errss.str());
  }

  /// Parse a string of the form z,y,x into a Index(z,y,x).If the parsing fails 
  /// an llvm fatal_error is issued and the program exits with an appropriate 
  /// message. This function is not meant to be used by library code but rather 
  /// only for testing purposes in Opt plugins.
  kerma::Index parseIndexOrExit(const std::string& IndexStr, const char *ErrorMsg="") {
    std::stringstream ss(IndexStr);
    std::vector<unsigned int> vals;
    try {
      while(ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        vals.push_back(std::stoul(substr, 0, 10));
      }
      if ( vals.size() != 3)
        throw;
    } catch ( ... ) {
      llvmExit(ErrorMsg);
    }
    kerma::Index res(vals[0], vals[1], vals[2]);
    return res;
  }
}
#endif // KERMA_OPT_PLUGIN

static RegisterPass<kerma::MaterializeIdxPass> RegisterLoopInfoTestPass(
        /* pass arg  */   "kerma-mi", 
        /* pass name */   "Materialize Block and Thread indices in kernels", 
        /* modifies CFG */ false, 
        /* analysis pass*/ true);

namespace kerma {

char MaterializeIdxPass::ID = 4;

#ifdef KERMA_OPT_PLUGIN
MaterializeIdxPass::MaterializeIdxPass() : MaterializeIdxPass( parseIndexOrExit(MIBlock.getValue(), "Invalid blockIdx"),
                                                               parseIndexOrExit(MIThread.getValue(), "Invalid threadIdx"))
{}
#else
MaterializeIdxPass::MaterializeIdxPass() : MaterializeIdxPass( Index::Zero, Index::Zero)
{}
#endif

MaterializeIdxPass::MaterializeIdxPass(const Index& BlockIdx, const Index& ThreadIdx)
: Block(BlockIdx), Thread(ThreadIdx), TargetKernelFun(nullptr), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

MaterializeIdxPass::MaterializeIdxPass(const Index& BlockIdx, const Index& ThreadIdx, llvm::Function &KernelF)
: Block(BlockIdx), Thread(ThreadIdx), TargetKernelFun(&KernelF), TargetKernelName(nullptr), llvm::FunctionPass(ID)
{}

MaterializeIdxPass::MaterializeIdxPass(const Index& BlockIdx, const Index& ThreadIdx, const char *KernelName)
: Block(BlockIdx), Thread(ThreadIdx), TargetKernelFun(nullptr), TargetKernelName(KernelName), llvm::FunctionPass(ID)
{}

bool MaterializeIdxPass::doInitialization(llvm::Module &M) {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "Materialize blockIdx to: " << this->Block << '\n';
  llvm::errs() << "Materialize Block to: " << this->Block << '\n';
  llvm::errs() << "Target kernel: " << (MITarget.getValue().empty()? "all" : MITarget.getValue()) << "\n\n";

  if ( !this->hasWork())
    llvm::errs() << "Nothing to do\n";
  if ( !MITarget.empty())
    this->TargetKernelName = MITarget.getValue().c_str();
#endif

  if ( this->hasWork()) {
    DetectKernelsPass DetectKernels;
    DetectKernels.runOnModule(M);
    DetectKernels.getKernels(this->Kernels);
  }

  return false;
}

bool MaterializeIdxPass::doFinalization(llvm::Module &M) {
  return false;
}

bool MaterializeIdxPass::hasTargetKernel() const {
  return this->TargetKernelFun || this->TargetKernelName;
}

bool MaterializeIdxPass::isKernel(llvm::Function &F) {
  return (std::find(this->Kernels.begin(), this->Kernels.end(), &F) != this->Kernels.end())
      || nvvm::isKernelFunction(F);
}

bool MaterializeIdxPass::hasWork() const {
  return this->Block && this->Thread;
}

bool isBlockIdxBuiltin(llvm::Function &F) {
  return llvm::demangle(F.getName()).find(nvvm::BlockIdx) != std::string::npos;
}

bool isThreadIdxBuiltin(llvm::Function &F) {
  return llvm::demangle(F.getName()).find(nvvm::ThreadIdx) != std::string::npos;
}

namespace {
  ConstantInt *createUnsignedInt(LLVMContext &context, unsigned int value) {
    auto *ty = IntegerType::get(context, 32);
    return ConstantInt::get(ty, value, false);
  }
}

bool MaterializeIdxPass::analyzeKernel(llvm::Function &F) const {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "Kerma-MaterializeIdx: B:" << Block << ", T:" << Thread << "\n";
  llvm::errs() << "--Analyzing: " << llvm::demangle(F.getName()) << '\n';
#endif

  unsigned int changes = 0;

  for ( auto &BB : F) {
    for ( auto &I : BB) {
      if ( auto *CI = dyn_cast<CallInst>(&I)) {
        
        auto *Callee = CI->getCalledFunction();
        auto DemangledCalleeName = llvm::demangle(Callee->getName());

        if ( !isBlockIdxBuiltin(*Callee) && !isThreadIdxBuiltin(*Callee))
          continue;

#ifdef KERMA_OPT_PLUGIN
        if ( CI->getDebugLoc())
          llvm::errs() << (isBlockIdxBuiltin(*Callee)? "  -block.idx" : "  -thread.idx") 
                    << " call at line " << CI->getDebugLoc().getLine();
        else
          llvm::errs() << (isBlockIdxBuiltin(*Callee)? "  -block.idx" : "  -thread.idx")
                    << " call: " << *CI; 
#endif

        if ( nvvm::BlockIdx.x == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.x));
        }
        else if ( nvvm::BlockIdx.y == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.y));
        }
        else if ( nvvm::BlockIdx.z == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Block.z));
        }
        else if ( nvvm::ThreadIdx.x == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Thread.x));
        }
        else if ( nvvm::ThreadIdx.y == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Thread.y));
        }
        else if ( nvvm::ThreadIdx.z == DemangledCalleeName ) {
          I.replaceAllUsesWith(createUnsignedInt(F.getContext(), this->Thread.z));
        }

#ifdef KERMA_OPT_PLUGIN
        llvm::errs() << " -> " << Callee->getNumUses() << " uses materialized\n";
#endif 
        changes += Callee->getNumUses();
      }
    }
  }

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "\n";
#endif 
  return changes;
}

bool MaterializeIdxPass::runOnFunction(llvm::Function &F) {
  if ( F.isDeclaration() || F.isIntrinsic())
    return false;
  if ( !this->hasWork() || !this->isKernel(F))
    return false;
  
  if ( this->hasTargetKernel()) {
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

std::unique_ptr<MaterializeIdxPass> createMaterializeIdxPass() {
  return std::make_unique<MaterializeIdxPass>();
}

std::unique_ptr<MaterializeIdxPass> 
createMaterializeIdxPass(const Index& Block, const Index& Thread) {
  return std::make_unique<MaterializeIdxPass>(Block, Thread);
}

std::unique_ptr<MaterializeIdxPass>
createMaterializeIdxPass(const Index& Block, const Index& Thread, llvm::Function &F) {
  return std::make_unique<MaterializeIdxPass>(Block, Thread, F);
}

std::unique_ptr<MaterializeIdxPass>
createMaterializeIdxPass(const Index& Block, const Index& Thread, const char *KernelName) {
  return std::make_unique<MaterializeIdxPass>(Block, Thread, KernelName);
}

} // namespace kerma