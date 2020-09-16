#include "kerma/Transforms/InstruMemOpPrintf.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/RT/Util.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Support/Parse.h"
#include "kerma/Transforms/LinkDeviceRT.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

#ifdef KERMA_OPT_PLUGIN

#include "llvm/Support/CommandLine.h"

namespace {

// Set up some cl args for Opt
cl::OptionCategory MOInstrOptionCategory("Kerma Instrument Memory Operations pass Options (--kerma-mo-instr)");
cl::opt<std::string> MOInstrTarget("mo-instr-target", cl::Optional, cl::desc("Target specific kernel function"),
                                    cl::value_desc("kernel_name[,kernel_name]"), cl::cat(MOInstrOptionCategory), cl::init(""));
cl::opt<bool> MOInstrReportNS("mo-instr-report-ns", cl::Optional, cl::desc("Report the namespace of the memory access"),
                              cl::cat(MOInstrOptionCategory), cl::init(false));
cl::opt<bool> MOInstrIgnoreLocal("mo-instr-ignore-local", cl::Optional, cl::desc("Ignore local memory accesses"),
                                 cl::cat(MOInstrOptionCategory), cl::init(false));
cl::opt<kerma::InstruMemOpPrintfPass::Op> MOInstrOp("mo-instr-op", cl::desc("Select ops to instrument"), cl::Optional,
                            cl::values(
                              clEnumValN(kerma::InstruMemOpPrintfPass::Op::Load, "load", "Instrument loads only"),
                              clEnumValN(kerma::InstruMemOpPrintfPass::Op::Store, "store", "Instrument stores only"),
                              clEnumValN(kerma::InstruMemOpPrintfPass::Op::All, "all", "Instrument both loads and stores (default)")
                            ), cl::init(kerma::InstruMemOpPrintfPass::Op::All), cl::cat(MOInstrOptionCategory));

}

#endif

static RegisterPass<kerma::InstruMemOpPrintfPass> RegisterMemOpInstrumentationPass(
        /* pass arg  */   "kerma-mo-instr",
        /* pass name */   "Instrument memory operations in CUDA kernels",
        /* modifies CFG */ false,
        /* analysis pass*/ true);

namespace kerma {



char InstruMemOpPrintfPass::ID = 4;

InstruMemOpPrintfPass::InstruMemOpPrintfPass(enum InstruMemOpPrintfPass::Op Op, bool IgnoreLocal)
: TargetOp(Op), IgnoreLocal(IgnoreLocal), ModulePass(ID) {}

InstruMemOpPrintfPass::InstruMemOpPrintfPass(const std::vector<std::string>& Targets, bool IgnoreLocal)
: InstruMemOpPrintfPass::InstruMemOpPrintfPass(Targets, InstruMemOpPrintfPass::Op::All, IgnoreLocal)
{}

InstruMemOpPrintfPass::InstruMemOpPrintfPass(const std::vector<std::string>& Targets,
                                                   enum InstruMemOpPrintfPass::Op Op, bool IgnoreLocal)
: TargetOp(Op), IgnoreLocal(IgnoreLocal), ModulePass(ID)
{
  for ( const auto& target : Targets)
    this->Targets.push_back(target);
}

bool InstruMemOpPrintfPass::hasTargetFunction() { return !this->Targets.empty(); }

InstruMemOpPrintfPass::Op InstruMemOpPrintfPass::getTargetOp() { return TargetOp; }

void InstruMemOpPrintfPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<DetectKernelsPass>();
}


std::pair<unsigned int, unsigned int> instrumentKernel(Function &F) {
#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "--Instrumenting " << demangle(F.getName().str()) << ": ";
#endif

  unsigned int Loads=0, Stores=0;

  for ( auto& BB : F) {
    for ( auto& I : BB) {

    }
  }



#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << Loads << " loads, " << Stores << " stores\n";
#endif
  return std::make_pair(Loads, Stores);
}

bool InstruMemOpPrintfPass::runOnModule(Module &M) {
  unsigned int changes = 0;

  if ( M.getTargetTriple().find("nvptx") == std::string::npos)
    return changes;

  if ( !KermaRTLinked(M)) {
#ifdef KERMA_OPT_PLUGIN
    llvm::report_fatal_error("KermaRT not found in " + M.getName());
#else
    LinkDeviceRTPass LinkKermaRTDevice;
    LinkKermaRTDevice.runOnModule(M);
  // Maybe we can programmatically link the the RT module here
  // https://github.com/UniHD-CEG/cuda-memtrace/blob/master/lib/LinkDeviceSupport.cpp

  //     throw rt::KermaRTNotFoundError(M.getName());
#endif
  }

#ifdef KERMA_OPT_PLUGIN
  this->TargetOp = MOInstrOp.getValue();
  this->IgnoreLocal = MOInstrIgnoreLocal.getValue();

  if ( !MOInstrTarget.getValue().empty()) {
    auto vals = parseDelimStr(MOInstrTarget, ',');
    for ( auto&& val : vals)
      Targets.push_back(val);
  }
#endif

  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();

  for ( auto* kernel : Kernels)
    if ( hasTargetFunction() ) {
      if ( std::find(Targets.begin(), Targets.end(), demangleFnWithoutArgs(*kernel)) != Targets.end()) {
        auto res = instrumentKernel(*kernel);
        changes += res.first + res.second;
      }
    } else {
      auto res = instrumentKernel(*kernel);
      changes += res.first + res.second;
    }

  return changes;
}

} // namespace kerma