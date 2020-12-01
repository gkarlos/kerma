#ifndef KERMA_TRANSFORMS_MATERIALIZER_H
#define KERMA_TRANSFORMS_MATERIALIZER_H

#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include <llvm-10/llvm/IR/Value.h>
#include <llvm/Pass.h>

namespace kerma {

/// This pass materializes the assumptions made for each kernel
/// It does so by replacing uses of IR values with the assumed
/// values.
/// By default, only grid/block dims are materialized
/// Materialization of args and indices are optional and controlled
/// by the third and fourth constructor arguments.
/// TODO: For now we only handle "local" assumptions. That is
///       grid/block dims, block/thread idx, and arguments for
///       each kernel. We should also handle global values
///       e.g __device__ int x
///       This is ok for now as the Assumptions parse also
///       ignores those at the moment. But we should fix it
class AssumptionMaterializerPass : public llvm::ModulePass {
public:
  static char ID;
  AssumptionMaterializerPass(KernelInfo &KI, AssumptionInfo &AI,
                             bool MaterializeDims = true,
                             bool MaterializeArgs = false);
  bool runOnModule(llvm::Module &M) override;
  // void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  llvm::StringRef getPassName() const override {
    return "AssumptionMaterializerPass";
  }

private:
  AssumptionInfo &AI;
  KernelInfo &KI;
  bool IgnoreDims;
  bool IgnoreArgs;
  bool IgnoreIdx;
};

void MaterializeArguments(Kernel &K, AssumptionInfo &AI);
void MaterializeBody(Kernel &K, AssumptionInfo &AI, const Index &Bid,
                     const Index &Tid);
void MaterializeDims(Kernel &K, AssumptionInfo &AI);
void MaterializeIdx(Kernel &K, AssumptionInfo &AI, const Index &Bid,
                    const Index &Tid);
/// Materializes the blockIdx.{z,y,x} values used in a kernel
/// IMPORTANT: The function does not perform any checks about
/// the legality of the index. It is up to the caller to make
/// sure that the index is valid within the grid.
void MaterializeBlockIdx(Kernel &K, const Index &I);
void MaterializeBlockIdx(llvm::Function *F, const Index &Idx);
void MaterializeBlockIdx(llvm::Function *F, llvm::Value *Vz, llvm::Value *Vy,
                         llvm::Value *Vx);
/// Materializes the threadIdx.{z,y,x} values used in a kernel
/// IMPORTANT: The function does not perform any checks about
/// the legality of the index. It is up to the caller to make
/// sure that the index is valid within the grid.
void MaterializeThreadIdx(Kernel &K, const Index &I);
void MaterializeThreadIdx(llvm::Function *F, const Index &Idx);
void MaterializeThreadIdx(llvm::Function *F, llvm::Value *Vz, llvm::Value *Vy,
                          llvm::Value *Vx);
} // namespace kerma

#endif // KERMA_TRANSFORMS_MATERIALIZER