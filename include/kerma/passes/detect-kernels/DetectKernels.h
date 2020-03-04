//
// Created by gkarlos on 1/3/20.
//

#ifndef KERMA_STATIC_ANALYSIS_DETECTKERNELS_H
#define KERMA_STATIC_ANALYSIS_DETECTKERNELS_H

#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <kerma/Cuda/CudaModule.h>
#include <kerma/Cuda/CudaKernel.h>

#include <set>

namespace kerma {

struct DetectKernelsPass : public llvm::ModulePass {
public:
  static char ID;
  explicit DetectKernelsPass(CudaModule *program = nullptr)
  : ModulePass(ID), program_(program)
  {}

  bool runOnModule(llvm::Module &M) override;
  bool doInitialization(llvm::Module& M) override;
  bool doFinalization(llvm::Module& M) override;
  virtual std::set<CudaKernel*>& getKernels();
  virtual void attachProgram(CudaModule *program);
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;

private:
  CudaModule* program_;
  std::set<CudaKernel*> kernels_;
};

} // NAMESPACE kerma
#endif // KERMA_STATIC_ANALYSIS_DETECTKERNELS_H
