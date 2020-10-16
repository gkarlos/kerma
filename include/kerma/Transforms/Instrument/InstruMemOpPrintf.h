#ifndef KERMA_TRANSFORMS_INSTRUMENT_INSTRU_MEM_OP_PRINTF_H
#define KERMA_TRANSFORMS_INSTRUMENT_INSTRU_MEM_OP_PRINTF_H

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>

#include <set>
#include <stdexcept>
#include <unordered_map>

namespace kerma {

enum MemOp { Load, Store, Atomic, All};

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

struct SourceLoc { unsigned int line; unsigned int col; };
enum Mode { BLOCK_MODE=0, WARP_MODE, THREAD_MODE };

class InstruMemOpPrintfPass : public llvm::ModulePass {
public:

  // enum Op { Load=1, Store, All };

private:
  std::vector<std::string> Targets;
  MemOp TargetOp;
  Mode Mode=BLOCK_MODE;
  bool IgnoreLocal;

private: /// per-run data structures. need to be cleared before each run.
  std::set<llvm::Function*> InstrumentedFunctions;
  std::unordered_map<std::string, llvm::Constant*> GlobalVariableForSymbol;

private:
  bool instrumentKernelMeta(llvm::Function &F, unsigned char id);
  bool instrumentBaseAddresses(llvm::Function &F, unsigned char id);
  bool instrumentGlobalBaseAddresses(llvm::Function &Kernel, unsigned char KernelId, llvm::Function &Hook);
  bool instrumentArgBaseAddresses(llvm::Function &Kernel, unsigned char KernelId, llvm::Function &Hook);
  bool instrumentAccess(llvm::Module *M, unsigned char KernelId, llvm::Value *Ptr, SourceLoc &Loc, llvm::Instruction *InsertBefore, MemOp op);

  PassStats instrumentKernel(llvm::Function &F, unsigned char id);

public:
  static char ID;
  InstruMemOpPrintfPass(MemOp Op=All, bool IgnoreLocal=true);
  InstruMemOpPrintfPass(const std::vector<std::string>& Targets, bool IgnoreLocal=true);
  InstruMemOpPrintfPass(const std::vector<std::string>& Targets, MemOp Op=All, bool IgnoreLocal=true);

public:
  bool runOnModule(llvm::Module &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  enum Mode getMode();
  bool hasTargetFunction();
  MemOp getTargetOp();
  bool WTF(int x, int y = 6);
  const std::string getOpString(MemOp op);
  const char getOpChar(MemOp op);
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_INSTRU_MEM_OP_PRINTF_H