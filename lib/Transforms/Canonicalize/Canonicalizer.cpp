#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/RT/Util.h"
#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"
#include "kerma/Transforms/Canonicalize/GepifyMem.h"
#include "kerma/Transforms/Canonicalize/SimplifyGEP.h"


#include <llvm/Support/FormatVariadic.h>

#include <numeric>

using namespace kerma;
using namespace llvm;

char CanonicalizerPass::ID = 111;

CanonicalizerPass::CanonicalizerPass() : ModulePass(ID) {}

void CanonicalizerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool CanonicalizerPass::runOnModule(llvm::Module& M) {

  bool changed = false;
  unsigned checked = 0;
  std::vector<bool> Changes(3, false);

  GepifyMemPass GepifyMem;
  BreakConstantGEPPass BreakConstantGEP;
  SimplifyGEPPass SimplifyGEP;

  for ( auto& F : M) {
    if ( F.isDeclaration() || F.isIntrinsic() || nvvm::isCudaAPIFunction(F) || isDeviceRTFunction(F))
      continue;

    ++checked;

    Changes[0] = GepifyMem.runOnFunction(F);
    Changes[1] = BreakConstantGEP.runOnFunction(F);
    Changes[2] = SimplifyGEP.runOnFunction(F);

    changed |= std::accumulate(Changes.begin(), Changes.end(), 0);
    std::fill(Changes.begin(), Changes.end(), false);
  }

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << '[' << formatv("{0,15}", "Canonicalizer") << "] Run on " << checked << " functions\n";
#endif

  return changed;
}

namespace {
static RegisterPass<CanonicalizerPass> RegisterCanonicalizerPass(
        /* pass arg  */   "kerma-canon",
        /* pass name */   "Canonicalize the IR",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}