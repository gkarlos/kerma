#ifndef KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H
#define KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H

#include "kerma/Base/Index.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Assumption.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Pass.h>
#include <unordered_map>

namespace kerma{

class AssumptionInfo {
  friend class DetectAsumptionsPass;

private:
  std::unordered_map<llvm::Value *, ValAssumption*> Vals;
  std::unordered_map<llvm::Value *, DimAssumption*> Dims;
  std::unordered_map<llvm::Function *, LaunchAssumption*> Launches;

  std::unordered_map<llvm::Function *, Index> BlockSelections;
  std::unordered_map<llvm::Function *, Index> ThreadSelections;
  std::unordered_map<llvm::Function *, unsigned> WarpSelections;

public:
  AssumptionInfo()=default;
  void setBlockSelection(llvm::Function *F, const Index &Idx) {
    BlockSelections[F] = Idx;
  }
  void setThreadSelection(llvm::Function *F, const Index &Idx) {
    ThreadSelections[F] = Idx;
  }
  void setWarpSelections(llvm::Function *F, unsigned Idx) {
    WarpSelections[F] = Idx;
  }

  /// Retrieve the selected block index for a kernel function
  /// If none exists index (0,0,0) is assigned to F and returned
  const Index &getBlockSelection(llvm::Function *F) {
    auto it = BlockSelections.find(F);
    if ( it == BlockSelections.end()) {
      BlockSelections[F] = Index(0,0,0);
      return BlockSelections[F];
    }
    return it->second;
  }

  /// Retrieve the selected thread index for a kernel function
  /// If none exists index (0,0,0) is assigned to F and returned
  const Index &getThreadSelection(llvm::Function *F) {
    auto it = ThreadSelections.find(F);
    if ( it == ThreadSelections.end()) {
      ThreadSelections[F] = Index(0,0,0);
      return BlockSelections[F];
    }
    return it->second;
  }

  /// Retrieve the selected thread index for a kernel function
  /// If none exists warp 0 is assigned to F and returned
  unsigned getWarpSelection(llvm::Function *F) {
    auto it = WarpSelections.find(F);
    if ( it == WarpSelections.end()) {
      ThreadSelections[F] = 0;
      return BlockSelections[F];
    }
    return it->second;
  }

  // ~AssumptionInfo() {
  //   for ( auto E: Vals)
  //     delete E.second;
  //   for ( auto E: Dims)
  //     delete E.second;
  //   for ( auto E: Launches)
  //     delete E.second;
  // }
  AssumptionInfo& add(llvm::Value *, Assumption *A);
  AssumptionInfo& addLaunch(llvm::Function *F, LaunchAssumption *LA);
  unsigned getSize() { return Vals.size() + Dims.size(); }
  unsigned getLaunchCount() { return Launches.size(); }
  unsigned getValCount() { return Vals.size(); }
  unsigned getDimCount() { return Dims.size(); }
  std::vector<ValAssumption*> getVals();
  std::vector<DimAssumption*> getDims();
  std::vector<LaunchAssumption*> getLaunches();
  std::vector<Assumption*> getAll();
  Assumption *getForArg(llvm::Argument *Arg) const {
    if ( Arg) {
      for ( auto &E : Vals)
        if ( E.first == Arg)
          return E.second;
      for ( auto &E : Dims)
        if ( E.first == Arg)
          return E.second;
      return nullptr;
    }
    return nullptr;
  }
  AssumptionInfo &operator=(const AssumptionInfo &O);

  LaunchAssumption *getLaunch(llvm::Function *F) const {
    if ( !F) return nullptr;
    if ( auto E = Launches.find(F); E != Launches.end())
      return E->second;
    return nullptr;
  }
  LaunchAssumption *getLaunch(Kernel &K) { return getLaunch(K.getFunction()); }
  void dump();
};

class DetectAsumptionsPass : public llvm::ModulePass {
public:
  static char ID;
#ifdef KERMA_OPT_PLUGIN
  DetectAsumptionsPass();
#endif
  DetectAsumptionsPass(KernelInfo *KI, MemoryInfo *MI);

  bool runOnModule(llvm::Module &M) override;
  AssumptionInfo &getAssumptionInfo() { return AI; }
  llvm::StringRef getPassName() const override { return "DetectAssumptionsPass"; }

private:
  KernelInfo *KI;
  MemoryInfo *MI;
  AssumptionInfo AI;
};


} // namespace kerma


#endif // KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H