#ifndef KERMA_PASS_MATERIALIZE_COMPUTE_UNIT_H
#define KERMA_PASS_MATERIALIZE_COMPUTE_UNIT_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "kerma/Base/Dim.h"

#include <memory>

namespace kerma {

/// 
class MaterializeGridPass : public llvm::FunctionPass {

public:
  static char ID;
  // This constructor is only meant to be used when the pass
  // is run with opt. i.e the values are retrieved through
  // the command line flags passed in opt. If no flags are
  // passed a passed created with this constructor will do
  // nothing
  MaterializeGridPass();

  MaterializeGridPass(const Dim& Grid, const Dim& Block);
  MaterializeGridPass(const Dim& Grid, const Dim& Block, llvm::Function &Kernel);
  MaterializeGridPass(const Dim& Grid, const Dim& Block, const char *KernelName);

public:
  bool doInitialization(llvm::Module &M) override;
  bool doFinalization(llvm::Module &M) override;
  bool runOnFunction(llvm::Function &F) override;
  virtual void print(llvm::raw_ostream &O, const llvm::Module *M) const override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool isTargeted();

private:
  bool analyzeKernel(llvm::Function &F) const;
  bool isKernel(llvm::Function &F);
  bool hasWork() const;

private:
  std::vector<llvm::Function*> Kernels;
  llvm::Function *TargetKernelFun;
  char *TargetKernelName;
  Dim Grid;
  Dim Block;
};

std::unique_ptr<MaterializeGridPass>
createMaterializeGridPass();

std::unique_ptr<MaterializeGridPass>
createMaterializeGridPass(const Dim& Grid, const Dim& Block);

std::unique_ptr<MaterializeGridPass>
createMaterializeGridPass(const Dim& Grid, const Dim& Block, llvm::Function &F);

} // namespace kerma

#endif