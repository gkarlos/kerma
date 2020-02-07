#ifndef KERMA_STATIC_ANALYSIS_DG_DGDOTPASS_H
#define KERMA_STATIC_ANALYSIS_DG_DGDOTPASS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <kerma/passes/dg/Dot.h>
#include <kerma/cuda/CudaKernel.h>

#include <set>

namespace kerma
{

class DGDotPass : public llvm::FunctionPass
{
public:
  static char ID;

  DGDotPass();
  ~DGDotPass()=default;

  std::set<DotNode>& getNodes();
  std::set<DotEdge>& getEdges();
  

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;

  friend class Dot;

private:
  std::set<DotNode> nodes_;
  std::set<DotEdge> edges_;
  std::vector<llvm::Value*> unused_;
  DotWriter dotWriter_;
  std::map<llvm::Value*, DotNode> lookup_;
  DotNode& lookupNodeOrNew(llvm::Value &V);
};

class DGDotKernelPass : public DGDotPass
{
public:
  static char ID;

  DGDotKernelPass();
  ~DGDotKernelPass()=default;

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;

private:
  std::set<cuda::CudaKernel*>* kernels_;
};

} /// NAMESPACE kerma

#endif /// KERMA_STATIC_ANALYSIS_DGDOTPASS_H