#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"



using namespace llvm;

namespace {
  struct KermaTestPass : public FunctionPass {
    static char ID;
    KermaTestPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) override {
      MDNode *meta = F.getMetadata("kernel");

      errs() << "Kerma: ";
      errs().write_escaped(F.getName()) << '\n';
      return false;
    }
  };
}

char KermaTestPass::ID = 0;

static RegisterPass<KermaTestPass> X("kerma", "Kerma Test Pass", false, false);

static void loadPass(const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
  KermaTestPass *p = new KermaTestPass();
  PM.add(p);
}

static RegisterStandardPasses RegisterDevicePass0(PassManagerBuilder::EP_OptimizerLast, loadPass);

