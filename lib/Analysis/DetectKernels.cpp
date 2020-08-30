#include "kerma/Analysis/DetectKernels.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <vector>


namespace kerma {

using namespace llvm;

char DetectKernelsPass::ID = 1;

DetectKernelsPass::DetectKernelsPass()
: llvm::ModulePass(ID)
{}

std::vector<Function*> DetectKernelsPass::getKernels() {
  std::vector<Function*> res(this->Kernels);
  return res;
}

void DetectKernelsPass::getKernels(std::vector<Function*> &Kernels) {
  for ( auto *F : this->Kernels)
    Kernels.push_back(F);
}

bool DetectKernelsPass::runOnModule(llvm::Module &M) {
  //TODO if host module do nothing
  if ( NamedMDNode *NVVMMD = M.getNamedMetadata("nvvm.annotations")) {
    for ( const MDNode *node : NVVMMD->operands()) {
      if ( ValueAsMetadata *VAM = dyn_cast_or_null<ValueAsMetadata>(node->getOperand(0).get())) {
        if ( Function *F = dyn_cast<Function>(VAM->getValue())) {
          // function with nvvm.annotation = kernel, 
          // but lets be more robust:
          //  check if Arg 1 of the MDNode is the string kernel
          if ( MDString *MDStr = dyn_cast_or_null<MDString>(node->getOperand(1).get())) {
            if ( MDStr->getString() == "kernel")
              this->Kernels.push_back(F);
          }
        }
      }
    }
  }
  return false;
}

/// This method is invoked when the pass is run in
/// opt with the -analyze flag passed
void DetectKernelsPass::print(llvm::raw_ostream &O, const Module *M) const {
  O << "Found " << this->Kernels.size() << (this->Kernels.size() > 1? " kernels" : "kernel");
  if ( this->Kernels.size()) {
    O << ":\n";
    for ( int i = 0; i < this->Kernels.size(); ++i)
      O << "  " << i << ". " << this->Kernels.at(i)->getName() << "\n";
  } else {
    O << "\n";
  }
}

void DetectKernelsPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

std::unique_ptr<DetectKernelsPass>
createDetectKernelsPass() {
  return std::make_unique<DetectKernelsPass>();
}

} // end namespace kerma

namespace {

static llvm::RegisterPass<kerma::DetectKernelsPass> RegisterDetectKernelsPass(/* pass arg  */    "kerma-detect-kernels", 
                                                                 /* pass name */    "Detect kernel functions", 
                                                                 /* modifies CFG */ false, 
                                                                 /* analysis pass*/ true);
}

