#include "kerma/Analysis/DetectMemoryAccesses.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/Stmt.h"
#include "kerma/Base/Node.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include <algorithm>
#include <cstddef>
#include <llvm-10/llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <memory>
#include <stack>
#include <utility>

namespace kerma {

using namespace llvm;
using namespace std;

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

std::vector<MemoryAccess>
MemoryAccessInfo::getAccessesForKernel(unsigned int ID) {
  std::vector<MemoryAccess> Res;
  auto itL = L.find(ID);
  auto itS = S.find(ID);
  auto itA = A.find(ID);
  auto itMM = MM.find(ID);
  auto itMC = MC.find(ID);
  auto itMS = MS.find(ID);

  Res.insert(Res.end(), L[ID].begin(), L[ID].end());
  Res.insert(Res.end(), S[ID].begin(), S[ID].end());
  Res.insert(Res.end(), A[ID].begin(), A[ID].end());
  Res.insert(Res.end(), MM[ID].begin(), MM[ID].end());
  Res.insert(Res.end(), MC[ID].begin(), MC[ID].end());
  Res.insert(Res.end(), MS[ID].begin(), MS[ID].end());
  return Res;
}

std::vector<std::pair<llvm::Instruction *, llvm::Value *>>
MemoryAccessInfo::getIgnoredAccessesForKernel(const Kernel &K) {
  std::vector<std::pair<llvm::Instruction *, llvm::Value *>> Res;
  Res.insert(Res.end(), IgnL[K.getID()].begin(), IgnL[K.getID()].end());
  Res.insert(Res.end(), IgnS[K.getID()].begin(), IgnS[K.getID()].end());
  Res.insert(Res.end(), IgnA[K.getID()].begin(), IgnA[K.getID()].end());
  Res.insert(Res.end(), IgnMM[K.getID()].begin(), IgnMM[K.getID()].end());
  Res.insert(Res.end(), IgnMC[K.getID()].begin(), IgnMC[K.getID()].end());
  Res.insert(Res.end(), IgnMS[K.getID()].begin(), IgnMS[K.getID()].end());
  return Res;
}

Stmt *MemoryAccessInfo::getStmtForAccess(const MemoryAccess &MA) {
  for (auto &Entry : MAS)
    for (auto &S : Entry.second)
      for (auto &A : S.getAccesses())
        if (A == MA)
          return &S;
  return nullptr;
}

Stmt *MemoryAccessInfo::getStmtAtRange(const SourceRange &R,
                                                   bool strict) {
  for (auto &Entry : MAS)
    for (auto &S : Entry.second) {
      if (S.getRange().contains(R))
        return &S;
      if (!strict && S.getRange().overlaps(R))
        return &S;
    }
  return nullptr;
}

unsigned int MemoryAccessInfo::getNumStmts() {
  unsigned int res = 0;
  for (auto &E : MAS)
    res += E.second.size();
  return res;
}

unsigned int MemoryAccessInfo::getNumAccesses() {
  unsigned int res = 0;
  for (auto &E : L)
    res += E.second.size();
  for (auto &E : S)
    res += E.second.size();
  for (auto &E : A)
    res += E.second.size();
  for (auto &E : MM)
    res += E.second.size();
  for (auto &E : MC)
    res += E.second.size();
  for (auto &E : MS)
    res += E.second.size();
  return res;
}

unsigned int MemoryAccessInfo::getNumIgnoredAccesses() {
  unsigned int res = 0;
  for (auto &E : IgnL)
    res += E.second.size();
  for (auto &E : IgnS)
    res += E.second.size();
  for (auto &E : IgnA)
    res += E.second.size();
  for (auto &E : IgnMM)
    res += E.second.size();
  for (auto &E : IgnMC)
    res += E.second.size();
  for (auto &E : IgnMS)
    res += E.second.size();
  return res;
}

// Pass

using namespace llvm;
using namespace kerma::nvvm;

char DetectMemoryAccessesPass::ID = 6;

DetectMemoryAccessesPass::DetectMemoryAccessesPass(KernelInfo &KI,
                                                   MemoryInfo &MI,
                                                   SourceInfo &SI)
    : KI(KI), MI(MI), SI(SI), ModulePass(ID) {}

void DetectMemoryAccessesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

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
        }
        else if (auto *SI = dyn_cast<StoreInst>(&I)) {
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
        }
        else if (auto *CI = dyn_cast<CallInst>(&I)) {
          // FIXME: Here we handle only atomic calls
          //        We should also handle memory intrinsics
          //        The most important one is memcpy because
          //        something like struct S s = structs[...]
          //        will produce:
          //          alocca s;
          //          ...
          //          memcpy(s, structs + ..., sizeof(struct S));
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
        // else if (auto *BI = dyn_cast<BranchInst>(&I)) {
        //   auto *IfRange = this->SI.getRangeForBranch(BI);
        //   if ( IfRange)
        //     this->SI.print(errs(), *IfRange) << " " << *BI << '\n';
        // }
      }
    }
  }
  buildMemoryAccessInfo();
  return false;
}

// static void getNodesForLoop(LoopNest *ME, vector<LoopNest*> &Loops,
//                             vector<KermaNode *> &Nodes,
//                             vector<Stmt *> &Unassigned) {
//   // the stmts before our body belong to our parent
//   auto itPre = Unassigned.begin();
//   while (itPre != Unassigned.end()) {
//     if ((*itPre)->getRange().getEnd() < ME->getRange().getStart()) {
//       if (!ME->getParent()) // we are at the top level
//         Nodes.push_back(*itPre);
//       else { // we have a parent
//         assert(isa<LoopNest>(ME->getParent()) && "Parent not a LoopNest!");
//         dyn_cast<LoopNest>(ME->getParent())->addChild(**itPre);
//       }
//       itPre = Unassigned.erase(itPre++);
//     } else {
//       ++itPre;
//     }
//   }

//   // insert ourselves to the node-list if we are top level
//   if ( !ME->getParent())
//     Nodes.push_back(ME);
//   else {
//     // we have a parent
//     // no need to assert as if something went wrong the previous assertion will fail
//     dyn_cast<LoopNest>(ME->getParent())->addChild(*ME);
//   }

//   // Go through the subloops
//   for (auto *SubLoop : ME->getLoop()->getSubLoops()) {
//     Loops.push_back(new LoopNest(SubLoop, ME));
//     getNodesForLoop(Loops.back(), Loops, Nodes, Unassigned);
//   }

//   // add any statements between the end of
//   // the last subloop and our end also to us
//   auto itPost = Unassigned.begin();
//   while (itPost != Unassigned.end()) {
//     if ( (*itPost)->getRange().getEnd() <= ME->getRange().getEnd()) {
//       ME->addChild(**itPost);
//       itPost = Unassigned.erase(itPost++);
//     } else {
//       ++itPost;
//     }
//   }
// }

static void groupMemoryAccessesToStmts(Kernel &Kernel, SourceInfo &SI, MemoryAccessInfo &MAI) {
  auto Ranges = SI.getRangesInRange(Kernel.getSourceRange());
  auto Accesses = MAI.getAccessesForKernel(Kernel);

  for (auto &Access : Accesses) {
    // if there is a Stmt for this access bail
    if (MAI.getStmtForAccess(Access))
      continue;

    // find the range of the source statement for this access
    if (auto Range = SI.getRangeForLoc(Access.getLoc())) {
      // if there is a Stmt with that range in MAS
      if (auto *stmt = MAI.getStmtAtRange(Range)) {
        // just append the access to that statement
        stmt->addMemoryAccess(Access, SI);
      } else {
        // otherwise create a new Stmt
        Stmt S(Range);
        S.addMemoryAccess(Access, SI);
        MAI.addStmtForKernel(Kernel, S);
      }
    } else {
      errs() << "**WARNING** No range for " << Access << '\n';
    }
  }
}

void DetectMemoryAccessesPass::buildMemoryAccessInfo() {
  for (auto &Kernel : KI.getKernels())
    groupMemoryAccessesToStmts(Kernel, SI, MAI);
}

} // namespace kerma