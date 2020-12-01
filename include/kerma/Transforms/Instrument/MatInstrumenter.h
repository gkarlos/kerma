#ifndef KERMA_TRANSFORMS_INSTRUMENT_MAT_INSTRUMENTER_H
#define KERMA_TRANSFORMS_INSTRUMENT_MAT_INSTRUMENTER_H

#include "kerma/Base/Kernel.h"
#include "kerma/Base/Mode.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/RT/Util.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Pass.h>

#include <set>
#include <stdexcept>
#include <unordered_map>

namespace kerma {

struct PassStats {
  std::vector<llvm::Instruction*> Failed;
  unsigned int Loads=0,
               Stores=0,
               Atomics=0,
               InstruLoads=0,
               InstruStores=0,
               InstruAtomics=0;
  bool changes = false;
  PassStats& operator<< (const PassStats& Other) {
    Loads += Other.Loads;
    Stores += Other.Stores;
    Atomics += Other.Atomics;
    InstruLoads += Other.InstruLoads;
    InstruStores += Other.InstruStores;
    InstruAtomics += Other.InstruAtomics;
    return *this;
  }
};

class MatInstrumenter : public llvm::ModulePass {
private:
  std::vector<std::string> Targets;
  Mode Mode;

private:
  KernelInfo &KI;
  bool NullOffsets;
  std::set<llvm::Function*> InstrumentedFunctions;
  std::unordered_map<std::string, llvm::Constant*> GlobalVariableForSymbol;

private:
  bool instrumentMeta(const Kernel& Kernel, llvm::Instruction *TraceStatus);
  bool instrumentAccesses(const Kernel& Kernel, llvm::Instruction *TraceStatus);
  bool instrumentAccess(const Kernel& Kernel, llvm::Instruction *I, llvm::Instruction *TraceStatus);
  unsigned int instrumentGlobalBaseAddresses(const Kernel& Kernel, llvm::Instruction *InsertAfter, llvm::Instruction *TraceStatus);
  unsigned int instrumentArgBaseAddresses(const Kernel& Kernel, llvm::Instruction *InsertAfter, llvm::Instruction *TraceStatus);

public:
  static char ID;
  MatInstrumenter(KernelInfo &KI, enum Mode=BLOCK, bool NullOffsets=false);

public:
  bool runOnModule(llvm::Module &M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  enum Mode getMode();
  bool hasTargetFunction();
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_INSTRUMENT_MAT_INSTRUMENTER_H