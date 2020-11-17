#ifndef KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H
#define KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Assumption.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Pass.h>
#include <unordered_map>

namespace kerma{

class AssumptionInfo {
  friend class DetectAsumptionsPass;

private:
  std::unordered_map<llvm::Value *, ValAssumption> Vals;
  std::unordered_map<llvm::Value *, DimAssumption> Dims;
  std::unordered_map<llvm::Function *, LaunchAssumption> Launches;

public:
  AssumptionInfo()=default;
  AssumptionInfo& add(llvm::Value *, Assumption &A);
  AssumptionInfo& addLaunch(llvm::Function *F, LaunchAssumption &LA);
  unsigned getSize() { return Vals.size() + Dims.size(); }
  unsigned getLaunchCount() { return Launches.size(); }
  unsigned getValCount() { return Vals.size(); }
  unsigned getDimCount() { return Dims.size(); }
  std::vector<ValAssumption*> getVals();
  std::vector<DimAssumption*> getDims();
  std::vector<LaunchAssumption*> getLaunches();
  std::vector<Assumption*> getAll();
  Assumption *getForArg(llvm::Argument *Arg);
  LaunchAssumption *getLaunch(llvm::Function *F) {
    if ( !F) return nullptr;
    if ( auto E = Launches.find(F); E != Launches.end())
      return &E->second;
    return nullptr;
  }
  LaunchAssumption *getLaunch(Kernel &K) { return getLaunch(K.getFunction()); }
};

class DetectAsumptionsPass : public llvm::ModulePass {
public:
  static char ID;
#ifdef KERMA_OPT_PLUGIN
  DetectAsumptionsPass();
#endif
  DetectAsumptionsPass(KernelInfo *KI, MemoryInfo *MI);

  bool runOnModule(llvm::Module &M) override;
  AssumptionInfo getAssumptionInfo() { return AI; }

private:
  KernelInfo *KI;
  MemoryInfo *MI;
  AssumptionInfo AI;
};


} // namespace kerma


#endif // KERMA_ANALYSIS_DETECT_ASSUMPTIONS_H