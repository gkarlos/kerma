#include "kerma/Analysis/DetectMemoryAccesses.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <utility>

namespace kerma {

using namespace llvm;

MemoryAccess *MemoryAccessInfo::getByID(unsigned ID) {
  for (auto &E : L)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  for (auto &E : S)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  for (auto &E : A)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  for (auto &E : MM)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  for (auto &E : MC)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  for (auto &E : MS)
    for (auto &M : E.second)
      if (M.getID() == ID)
        return &M;
  return nullptr;
}

std::vector<MemoryAccess> MemoryAccessInfo::getForKernel(const Kernel &K) {
  std::vector<MemoryAccess> Res;
  auto itL = L.find(K.getID());
  auto itS = S.find(K.getID());
  auto itA = A.find(K.getID());
  auto itMM = MM.find(K.getID());
  auto itMC = MC.find(K.getID());
  auto itMS = MS.find(K.getID());

  Res.insert(Res.end(), itL->second.begin(), itL->second.end());
  Res.insert(Res.end(), itS->second.begin(), itS->second.end());
  Res.insert(Res.end(), itA->second.begin(), itA->second.end());
  Res.insert(Res.end(), itMM->second.begin(), itMM->second.end());
  Res.insert(Res.end(), itMC->second.begin(), itMC->second.end());
  Res.insert(Res.end(), itMS->second.begin(), itMS->second.end());
  return Res;
}

std::vector<std::pair<llvm::Instruction *, llvm::Value *>>
MemoryAccessInfo::getIgnoredForKernel(const Kernel &K) {
  std::vector<std::pair<llvm::Instruction *, llvm::Value *>> Res;
  Res.insert(Res.end(), IgnL[K.getID()].begin(), IgnL[K.getID()].end());
  Res.insert(Res.end(), IgnS[K.getID()].begin(), IgnS[K.getID()].end());
  Res.insert(Res.end(), IgnA[K.getID()].begin(), IgnA[K.getID()].end());
  Res.insert(Res.end(), IgnMM[K.getID()].begin(), IgnMM[K.getID()].end());
  Res.insert(Res.end(), IgnMC[K.getID()].begin(), IgnMC[K.getID()].end());
  Res.insert(Res.end(), IgnMS[K.getID()].begin(), IgnMS[K.getID()].end());
  return Res;
}

// Pass

using namespace llvm;
using namespace kerma::nvvm;

char DetectMemoryAccessesPass::ID = 6;

DetectMemoryAccessesPass::DetectMemoryAccessesPass(KernelInfo &KI,
                                                   MemoryInfo &MI)
    : KI(KI), MI(MI), ModulePass(ID) {}

bool DetectMemoryAccessesPass::runOnModule(Module &M) {

  for (auto &Kernel : KI.getKernels()) {
    for (auto &BB : *Kernel.getFunction()) {
      for (auto &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          auto *Obj =
              GetUnderlyingObject(LI->getPointerOperand(), M.getDataLayout());
          if (auto *M = MI.getMemoryForVal(Obj, &Kernel)) {
            MemoryAccess MA(*M, LI, LI->getPointerOperand(),
                            MemoryAccess::Load);
            MA.setLoc(SourceLoc::from(LI->getDebugLoc()));
            MAI.L[Kernel.getID()].push_back(MA);
          } else {
            MAI.IgnL[Kernel.getID()].push_back(std::make_pair(LI, Obj));
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          auto *Obj =
              GetUnderlyingObject(SI->getPointerOperand(), M.getDataLayout());
          if (auto *M = MI.getMemoryForVal(Obj, &Kernel)) {
            MemoryAccess MA(*M, SI, SI->getPointerOperand(),
                            MemoryAccess::Store);
            MA.setLoc(SourceLoc::from(SI->getDebugLoc()));
            MAI.S[Kernel.getID()].push_back(MA);
          } else {
            MAI.IgnS[Kernel.getID()].push_back(std::make_pair(SI, Obj));
          }
        } else if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (isAtomicFunction(*CI->getCalledFunction())) {
            auto *Obj =
                GetUnderlyingObject(CI->getArgOperand(0), M.getDataLayout());
            if (auto *M = MI.getMemoryForVal(Obj, &Kernel)) {
              MemoryAccess MA(*M, CI, CI->getArgOperand(0),
                              MemoryAccess::Store);
              MA.setLoc(SourceLoc::from(CI->getDebugLoc()));
              MAI.A[Kernel.getID()].push_back(MA);
            } else {
              MAI.IgnA[Kernel.getID()].push_back(std::make_pair(CI, Obj));
            }
          }
        }
      }
    }
  }

  return false;
}

} // namespace kerma