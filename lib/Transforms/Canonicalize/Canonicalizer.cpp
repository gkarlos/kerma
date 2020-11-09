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
#include <llvm/Transforms/Utils.h>

#include <memory>
#include <numeric>

using namespace kerma;
using namespace llvm;

char CanonicalizerPass::ID = 111;

CanonicalizerPass::CanonicalizerPass() : ModulePass(ID) {}

void CanonicalizerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
#ifdef KERMA_OPT_PLUGIN
  AU.addRequired<DetectKernelsPass>();
#endif
  AU.setPreservesAll();
}

bool CanonicalizerPass::runOnModule(llvm::Module &M) {

  bool Changed = false;
  unsigned Checked = 0;
  std::vector<bool> Changes(4, false);

  DeviceFunctionInliner Inliner;
  Changed = Inliner.runOnModule(M);

  GepifyMemPass GepifyMem;
  BreakConstantGEPPass BreakConstantGEP;
  SimplifyGEPPass SimplifyGEP;

#ifndef KERMA_OPT_PLUGIN
#endif

#ifdef KERMA_OPT_PLUGIN
  WithColor(errs(), HighlightColor::Note) << '[';
  WithColor(errs(), HighlightColor::String)
      << formatv("{0,15}", "Canonicalizer");
  WithColor(errs(), HighlightColor::Note) << "]\n";
#endif

#ifdef KERMA_OPT_PLUGIN
  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();
#else
  auto Kernels = getKernels(M);
#endif

  for (auto &K : Kernels) {
    ++Checked;
    Changes[1] = GepifyMem.runOnFunction(*K.getFunction());
    Changes[2] = BreakConstantGEP.runOnFunction(*K.getFunction());
    Changes[3] = SimplifyGEP.runOnFunction(*K.getFunction());

    Checked |= std::accumulate(Changes.begin(), Changes.end(), 0);
    std::fill(Changes.begin(), Changes.end(), false);
  }
  return Changed;
}

namespace {
static RegisterPass<CanonicalizerPass> RegisterCanonicalizerPass(
    /* pass arg  */ "kerma-canon",
    /* pass name */ "Canonicalize the IR",
    /* modifies CFG */ false,
    /* analysis pass*/ false);
}