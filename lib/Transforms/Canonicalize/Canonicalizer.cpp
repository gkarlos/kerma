#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/RT/Util.h"
#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"
#include "kerma/Transforms/Canonicalize/DeviceFunctionInliner.h"
#include "kerma/Transforms/Canonicalize/GepifyMem.h"
#include "kerma/Transforms/Canonicalize/SimplifyGEP.h"


#include <llvm/Demangle/Demangle.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/WithColor.h>

#include <numeric>

using namespace kerma;
using namespace llvm;

char CanonicalizerPass::ID = 111;

CanonicalizerPass::CanonicalizerPass() : ModulePass(ID) {}

void CanonicalizerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool CanonicalizerPass::runOnModule(llvm::Module& M) {

  bool Changed = false;
  unsigned Checked = 0;
  std::vector<bool> Changes(3, false);

  DeviceFunctionInliner Inliner;
  Changed = Inliner.runOnModule(M);

  GepifyMemPass GepifyMem;
  BreakConstantGEPPass BreakConstantGEP;
  SimplifyGEPPass SimplifyGEP;

#ifdef KERMA_OPT_PLUGIN
  WithColor(errs(), HighlightColor::Note) << '[';
  WithColor(errs(), HighlightColor::String) << formatv("{0,15}", "Canonicalizer");
  WithColor(errs(), HighlightColor::Note) << "]\n";
#endif

  for ( auto& F : M) {
    if ( F.isDeclaration() || F.isIntrinsic()
                           || nvvm::isCudaAPIFunction(F)
                           || nvvm::isAtomicFunction(F)
                           || nvvm::isReadOnlyCacheFunction(F)
                           || isDeviceRTFunction(F))
      continue;

    ++Checked;

    Changes[0] = GepifyMem.runOnFunction(F);
    Changes[1] = BreakConstantGEP.runOnFunction(F);
    Changes[2] = SimplifyGEP.runOnFunction(F);

    Checked |= std::accumulate(Changes.begin(), Changes.end(), 0);
    std::fill(Changes.begin(), Changes.end(), false);
  }
  return Changed;
}

namespace {
static RegisterPass<CanonicalizerPass> RegisterCanonicalizerPass(
        /* pass arg  */   "kerma-canon",
        /* pass name */   "Canonicalize the IR",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}