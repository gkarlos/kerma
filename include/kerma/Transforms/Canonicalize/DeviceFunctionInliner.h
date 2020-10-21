#ifndef KERMA_TRANSFORMS_CANONICALIZE_DEVICE_FUNCTION_INLINER_H
#define KERMA_TRANSFORMS_CANONICALIZE_DEVICE_FUNCTION_INLINER_H

#include <llvm/IR/Instruction.h>
#include <llvm/Pass.h>
namespace kerma {

class InlineInfo {
private:
  llvm::Function *InlinedFunction;
  llvm::Function *Caller;
  llvm::Instruction *PrevInst;
  llvm::Instruction *NextInst;

public:
  InlineInfo(llvm::Function *InlinedFunction, llvm::Function *Caller, llvm::Instruction *Prev, llvm::Instruction *Next)
  : InlinedFunction(InlinedFunction), Caller(Caller), PrevInst(Prev), NextInst(Next)
  {}
  bool operator==(const InlineInfo& Other) const {
    return InlinedFunction == Other.InlinedFunction
        && Caller == Other.Caller
        && PrevInst == Other.PrevInst
        && NextInst == Other.NextInst;
  }
};

/// This pass inlines every function call on device code
class DeviceFunctionInliner : public llvm::ModulePass {
private:
  unsigned int CallsChecked;
  unsigned int CallsInlined;
  unsigned int FunctionsMarkedForInlining;
  llvm::SmallVector<InlineInfo, 32> Info;
  bool doInline(llvm::Module& M);

public:
  static char ID;
  DeviceFunctionInliner();
  bool runOnModule(llvm::Module& M) override;

unsigned int getNumCallsChecked() { return CallsChecked; }
unsigned int getNumCallsInlined() { return CallsInlined; }

llvm::SmallVector<InlineInfo, 32> getInfo() { return Info; }

};

} // namespace kerma

#endif // KERMA_TRANSFORMS_CANONICALIZE_DEVICE_FUNCTION_INLINER_H