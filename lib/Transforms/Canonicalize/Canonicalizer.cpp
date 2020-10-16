#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"
#include "kerma/Transforms/Canonicalize/GepifyMem.h"
#include "kerma/Transforms/Canonicalize/SimplifyGEP.h"

#include <numeric>

using namespace kerma;
using namespace llvm;

char CanonicalizerPass::ID = 111;

CanonicalizerPass::CanonicalizerPass() : FunctionPass(ID) {}

bool CanonicalizerPass::runOnFunction(llvm::Function& F) {
  if ( F.isDeclaration() || F.isIntrinsic() || F.isDebugInfoForProfiling())
    return false;

  std::vector<bool> Changes(3, 0);

  GepifyMemPass GepifyMem;
  BreakConstantGEPPass BreakConstantGEP;
  SimplifyGEPPass SimplifyGEP;

  Changes[0] = GepifyMem.runOnFunction(F);
  Changes[1] = BreakConstantGEP.runOnFunction(F);
  Changes[2] = SimplifyGEP.runOnFunction(F);

  return std::accumulate(Changes.begin(), Changes.end(), 0);
}

namespace {
static RegisterPass<CanonicalizerPass> RegisterCanonicalizerPass(
        /* pass arg  */   "kerma-canon",
        /* pass name */   "Canonicalize the IR",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}