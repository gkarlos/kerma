#include <llvm/Pass.h>
#include "llvm/Demangle/Demangle.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include <llvm/Support/Debug.h>
#include <llvm/IR/DebugLoc.h>

#include <string>
#include <map>

#include <kerma/Cuda/NVVM.h>
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Cuda/CudaModule.h>
#include <kerma/Pass/DetectKernels.h>
#include <kerma/Support/LLVMStringUtils.h>
#include <kerma/Support/LLVMFunctionShorthands.h>

using namespace llvm;
using namespace kerma;

namespace kerma {
//  cl::opt<bool> Json("kerma-json", cl::desc("Write results to a json file"));
  // https://github.com/trailofbits/KRFAnalysis/blob/master/KRFAnalysisPass/KRF.cpp

DetectKernelsPass::DetectKernelsPass() 
: llvm::ModulePass(ID),
  cudaModule_(nullptr)
{}

DetectKernelsPass::DetectKernelsPass(CudaModule &program)
: llvm::ModulePass(ID),
  cudaModule_(&program)
{}

// void
// DetectKernelsPass::attachProgram(CudaModule *program)
// {
//   this->program_ = program;
//   // Try to populate the new program with kernels just in case the
//   // function is called after the pass has finished
//   if ( this->program_ != nullptr) {
//     for ( auto kernel : this->kernels_) {
//       this->program_->addKernel(kernel);
//     }
//   }
// }

bool
DetectKernelsPass::doInitialization(Module &)
{
  // Clear the kernels found, just in case the pass is re-run
  this->kernels_.clear();
  if ( this->cudaModule_ != nullptr)
    this->cudaModule_->getKernels().clear();
  return false;
}

bool
DetectKernelsPass::doFinalization(Module& M)
{
  // If there is a program attached to this pass, populate it with kernels
  if ( this->cudaModule_ != nullptr) {
    for ( auto kernel : this->kernels_)
      this->cudaModule_->addKernel(kernel);
  }
  return false;
}


void
DetectKernelsPass::print(llvm::raw_ostream &OS, const llvm::Module *M) const
{
  if ( kernels_.empty())
    errs() << "No kernels detected\n";
  else
    for ( auto kernel : this->kernels_)
      kernel.pp(OS);
}


bool
DetectKernelsPass::runOnModule(Module &M) {
  NamedMDNode *kernelMD = M.getNamedMetadata("nvvm.annotations");
  if ( kernelMD ) {
    for ( const llvm::MDNode *node : kernelMD->operands()) {
      Metadata *mdOperand = node->getOperand(0).get();
      if ( auto *v = dyn_cast_or_null<ValueAsMetadata>(mdOperand)) {
        if (auto *fun = dyn_cast<Function>(v->getValue())) {
          // nvvm.annotation + function = kernel but lets be more robust
          mdOperand = node->getOperand(1).get();
          // Check if the MDNode operand is a string and has the value "kernel"
          if (auto *mdStr = dyn_cast_or_null<MDString>(mdOperand)) {
            if (mdStr->getString() == "kernel") {
              CudaKernel kernel(*fun, getIRModuleSide(M));

              // Get line numbers
              if ( auto *DISub = kernel.getFn().getSubprogram() ) {
                kernel.setSignatureLineStart(DISub->getLine());
                kernel.setSignatureLineEnd(DISub->getScopeLine() - (DISub->getScopeLine() > DISub->getLine()? -1 : 0));
                kernel.setBodyLineStart(DISub->getScopeLine());
                
                unsigned int lastInstructionSourceCodeLine = 0;

                for ( auto &BB : kernel.getFn()) {
                  for ( auto &I : BB) {
                    if ( I.getDebugLoc() && I.getDebugLoc().getLine() > lastInstructionSourceCodeLine)
                      lastInstructionSourceCodeLine = I.getDebugLoc().getLine();
                  }
                }

                kernel.setBodyLineEnd(lastInstructionSourceCodeLine);
              }

              this->kernels_.insert(kernel);
            }
          }
        }
      }
    }
  }
  return false;
}

void
DetectKernelsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}


/// API

std::set<CudaKernel>
DetectKernelsPass::getKernels() {
  std::set<CudaKernel> res;
  res.insert(kernels_.begin(), kernels_.end());
  return res;
}

void
DetectKernelsPass::getKernels(std::set<CudaKernel> &kernels)
{
  for ( auto kernel : kernels_)
    kernels.insert(kernel);
}

bool
DetectKernelsPass::isKernel(llvm::Function &F)
{
  for ( auto kernel : kernels_)
    if ( static_cast<llvm::Value*>(&kernel.getFn()) == static_cast<llvm::Value*>(&F))
      return true;
  return false;
}

bool
DetectKernelsPass::hasCudaModuleAttached()
{
  return cudaModule_ != nullptr;
}

CudaModule *
DetectKernelsPass::getCudaModule()
{
  return cudaModule_;
}

bool
DetectKernelsPass::attachCudaModule(CudaModule &program)
{
  if ( cudaModule_ != nullptr)
    return false;
  
  cudaModule_ = &program;

  for ( auto kernel : kernels_)
    cudaModule_->addKernel(kernel);
  
  return true;
}

} /// NAMESPACE kerma


char kerma::DetectKernelsPass::ID = 1;

static RegisterPass<kerma::DetectKernelsPass> Y("kerma-detect-kernels", "Detect kernel functions", false, true);

// static void loadDetectModulesPass(const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
//   DetectModulesPass *p = new DetectModulesPass();
//   PM.add(p);
// }

// static RegisterStandardPasses RegisterDevicePass1(PassManagerBuilder::EP_EarlyAsPossible, loadDetectModulesPass);

