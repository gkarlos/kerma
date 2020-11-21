#include "kerma/Analysis/DetectKernels.h"

#include "kerma/Base/Kernel.h"
#include "kerma/NVVM/NVVMUtilities.h"

#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/WithColor.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>


namespace kerma {

using namespace llvm;

Kernel *KernelInfo::findByID(unsigned int ID) {
  for ( auto &K : Kernels)
    if ( K.getID() == ID)
      return &K;
  return nullptr;
}

Kernel *KernelInfo::find(llvm::Function *F) {
  if ( F) {
    for ( auto &K : Kernels)
      if ( K.getFunction() == F)
        return &K;
  }
  return nullptr;
}

bool KernelInfo::isKernel(Function &F) {
  for ( auto &K : Kernels) {
    if ( K.getFunction() == &F)
      return true;
  }
  return false;
}


// Pass

using namespace llvm;

char DetectKernelsPass::ID = 1;

DetectKernelsPass::DetectKernelsPass()
: llvm::ModulePass(ID)
{}

const std::vector<Kernel>& DetectKernelsPass::getKernels() {
  return Kernels;
}

void DetectKernelsPass::getKernels(std::vector<Kernel>& Kernels) {
  for ( auto &Kernel : this->Kernels)
    Kernels.push_back(Kernel);
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
              Kernels.push_back( Kernel(F));
          }
        }
      }
    }
  }

  WithColor(errs(), HighlightColor::Note) << '[';
  WithColor(errs(), raw_ostream::Colors::GREEN) << formatv("{0,15}", "KernelDetector");
  WithColor(errs(), HighlightColor::Note) << ']';
  errs() << ' ' << Kernels.size() << " kernels\n";

  return false;
}

/// This method is invoked when the pass is run in
/// opt with the -analyze flag passed
void DetectKernelsPass::print(llvm::raw_ostream &O, const Module *M) const {
  O << "Found " << this->Kernels.size() << (this->Kernels.size() > 1? " kernels" : "kernel");
  if ( this->Kernels.size()) {
    O << ":\n";
    for ( int i = 0; i < this->Kernels.size(); ++i)
      O << "  " << i << ". " << this->Kernels.at(i).getDemangledName() << "\n";
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

namespace {

static llvm::RegisterPass<kerma::DetectKernelsPass> RegisterDetectKernelsPass(
                                /* pass arg  */    "kerma-detect-kernels",
                                /* pass name */    "Detect kernel functions",
                                /* modifies CFG */ false,
                                /* analysis pass*/ true);
}


static std::map<const Module*, std::vector<Kernel>> KernelCache;

static void findKernels(const llvm::Module &M) {
  // First clear the cache
  if ( auto Entry = KernelCache.find(&M); Entry != KernelCache.end())
    Entry->second.clear();
  else
    KernelCache.insert(std::make_pair(&M, std::vector<Kernel>()));

  std::vector<Kernel>& Kernels = KernelCache[&M];

  if (NamedMDNode * NVVMMD = M.getNamedMetadata("nvvm.annotations")) {
    for (const MDNode *MDNode : NVVMMD->operands()) {
      for (const MDOperand &Op : MDNode->operands()) {
        Metadata * MD = Op.get();
        if ( ValueAsMetadata *VAMD = dyn_cast_or_null<ValueAsMetadata>(MD) )
          if ( Function *F = dyn_cast<Function>(VAMD->getValue()) )
            Kernels.push_back(Kernel(F));
      }
    }
  }
}

std::vector<Function*> getKernelFunctions(const Module &M, bool invalidateCacheEntry) {

  if ( invalidateCacheEntry || KernelCache.find(&M) == KernelCache.end())
    findKernels(M);

  auto& Kernels = KernelCache[&M];

  std::vector<Function*> Res;

  for ( auto &Kernel : Kernels)
    Res.push_back(Kernel.getFunction());

  return Res;
}

const std::vector<Kernel> & getKernels(const Module &M, bool invalidateCacheEntry) {
  if ( invalidateCacheEntry || KernelCache.find(&M) == KernelCache.end())
    findKernels(M);
  return KernelCache[&M];
}

bool isKerneFunction(const llvm::Function &F) {
  return nvvm::isKernelFunction(F);
}

} // end namespace kerma


