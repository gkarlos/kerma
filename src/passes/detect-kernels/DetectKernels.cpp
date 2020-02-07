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

#include <string>
#include <map>

#include <kerma/cuda/CudaSupport.h>
#include <kerma/passes/Util.h>
#include <kerma/passes/detect-kernels/DetectKernels.h>
#include <kerma/Support/LLVMStringUtils.h>

using namespace llvm;
using namespace kerma::cuda;

namespace kerma {
//  cl::opt<bool> Json("kerma-json", cl::desc("Write results to a json file"));
  // https://github.com/trailofbits/KRFAnalysis/blob/master/KRFAnalysisPass/KRF.cpp

std::set<CudaKernel*>&
DetectKernelsPass::getKernels() {
  return this->kernels_;
}

void
DetectKernelsPass::attachProgram(cuda::CudaProgram *program)
{
  this->program_ = program;
  // Try to populate the new program with kernels just in case the
  // function is called after the pass has finished
  if ( this->program_ != nullptr) {
    for ( auto kernel : this->kernels_) {
      this->program_->addKernel(kernel);
    }
  }
}

bool
DetectKernelsPass::doInitialization(Module &)
{
  // Clear the kernels found, just in case the pass is re-run
  this->kernels_.clear();
  if ( this->program_ != nullptr)
    this->program_->getKernels().clear();
  return false;
}

bool
DetectKernelsPass::doFinalization(Module& M)
{
  // If there is a program attached to this pass, populate it with kernels
  if ( this->program_ != nullptr) {
    for ( auto kernel : this->kernels_) {
      this->program_->addKernel(kernel);
    }
  }
  return false;
}


void
DetectKernelsPass::print(llvm::raw_ostream &OS, const llvm::Module *M) const
{
  if ( this->kernels_.empty())
    errs() << "No kernels detected\n";
  else
    for ( auto kernel : this->kernels_)
      kernel->pp(OS);
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
          if (auto *mdStr = dyn_cast_or_null<MDString>(mdOperand))
            if (mdStr->getString() == "kernel")
              this->kernels_.insert(
                  new CudaKernel(fun, getIRModuleSide(M)));
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

} /// NAMESPACE kerma


char kerma::DetectKernelsPass::ID = 1;

static RegisterPass<kerma::DetectKernelsPass> Y("kerma-detect-kernels", "Detect kernel functions", false, true);

// static void loadDetectModulesPass(const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
//   DetectModulesPass *p = new DetectModulesPass();
//   PM.add(p);
// }

// static RegisterStandardPasses RegisterDevicePass1(PassManagerBuilder::EP_EarlyAsPossible, loadDetectModulesPass);

