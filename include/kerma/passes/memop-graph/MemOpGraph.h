//
// Created by gkarlos on 1/5/20.
//
#ifndef KERMA_STATIC_ANALYSIS_MEMOPDEPS_H
#define KERMA_STATIC_ANALYSIS_MEMOPDEPS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include <kerma/cuda/CudaKernel.h>
#include <set>

using namespace llvm;

namespace kerma {

enum class MemOp { Unknown, LOAD, STORE };

enum class UnOp { Unknown, NEG /* - */, LNEG /* ! */, BNEG  /* ~ */ };

enum class BinOp {
  Unknown,
  ADD /* +  */, SUB /* -  */, MUL /* * */, DIV  /* /  */, MOD  /* % */,
  LOR /* |  */, BOR /* || */, XOR /* ^ */, LAND /* && */, BAND /* & */,
  SHL /* << */, SHR /* >> */
};


class MODep {
public:
  MODep(MemOp op) : op_(op)
  {}
  
  Value* getValue() { return this->value_; }
  MemOp getOp() { return this->op_; }

private:
  MemOp op_;
  Value* value_;
};

class MemOpGraphPass : public ModulePass {
public:
  MemOpGraphPass() : ModulePass(ID)
  {}
  static char ID;
  bool runOnModule(Module& M) override;
  void print(llvm::raw_ostream &OS, const llvm::Module *M) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  std::set<LoadInst*> allLoads_;
  std::set<StoreInst*> allStores_;
  virtual void analyzeKernel(cuda::CudaKernel *kernel);
};

} /// NAMESPACE kerma

#endif // KERMA_STATIC_ANALYSIS_MEMOPDEPS_H
