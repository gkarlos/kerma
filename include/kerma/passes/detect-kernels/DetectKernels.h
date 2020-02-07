//
// Created by gkarlos on 1/3/20.
//

#ifndef KERMA_STATIC_ANALYSIS_DETECTKERNELS_H
#define KERMA_STATIC_ANALYSIS_DETECTKERNELS_H

#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <kerma/cuda/CudaProgram.h>
#include <kerma/cuda/CudaKernel.h>

#include <set>

namespace kerma {

struct DetectKernelsPass : public llvm::ModulePass {
public:
  static char ID;
  explicit DetectKernelsPass(cuda::CudaProgram *program = nullptr)
  : ModulePass(ID), program_(program)
  {}

  bool runOnModule(llvm::Module &M) override;
  bool doInitialization(llvm::Module& M) override;
  bool doFinalization(llvm::Module& M) override;
  virtual std::set<cuda::CudaKernel*>& getKernels();
  virtual void attachProgram(cuda::CudaProgram *program);
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;

private:
  cuda::CudaProgram* program_;
  std::set<cuda::CudaKernel*> kernels_;
};

} // NAMESPACE kerma
#endif // KERMA_STATIC_ANALYSIS_DETECTKERNELS_H
