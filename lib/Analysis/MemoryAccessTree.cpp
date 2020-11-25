#include "kerma/Analysis/MemoryAccessTree.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemoryAccesses.h"
#include "kerma/Base/If.h"
#include "kerma/Base/Kernel.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/Memory.h"
#include "kerma/Base/Node.h"
#include <algorithm>
#include <cstdio>
#include <llvm-10/llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Instructions.h>

using namespace llvm;

namespace kerma {

void MemoryAccessTree::dump() {
  errs() << Kernel << '\n';
  for ( auto *Node : Tree)
    errs() << *Node << '\n';
  errs() << '\n';
}

void MemoryAccessTree::print(raw_ostream &OS) const {
  OS << Kernel << '\n';
  for ( auto *Node : Tree)
    OS << *Node << '\n';
  OS << '\n';
}


// Pass

char MATBuilder::ID = 43;

MATBuilder::MATBuilder(KernelInfo &KI, MemoryInfo &MI, SourceInfo &SI)
    : ModulePass(ID), KI(KI), MI(MI), SI(SI) {}

void MATBuilder::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<DetectMemoryAccessesPass>();
  // AU.addRequired<SimplifyCFGPass>();
  AU.setPreservesAll();
}

void MATBuilder::InsertNodeToParent(KermaNode *Node, KermaNode *Parent,
                                    Kernel &K) {
  if (Parent) {
    if (auto *IFParent = dyn_cast<If>(Parent)) {
      if (IFParent->getThenRange().contains(Node->getRange().getEnd()))
        IFParent->addThenChild(Node);
      else if (IFParent->getElseRange().contains(Node->getRange().getEnd()))
        IFParent->addElseChild(Node);
      else {
        if (IFParent->getCondRange().contains(Node->getRange().getEnd())) {
          if (auto *S = dyn_cast<Stmt>(Node)) {
            IFParent->setCond(dyn_cast<Stmt>(Node));
          } else {
            errs() << "ERROR: If condition is not a Stmt: " << Node->getID()
                   << " - " << Node->getRange() << " - " << Node->getKind()
                   << '\n';
          }
        } else
          errs() << "ERROR: Parent is an if but ranges don't match!\n";
      }
    } else if (auto *LoopParent = dyn_cast<LoopNest>(Parent)) {
      if ( SI.getForInitRangeForLoc(Node->getRange().getStart()))
        LoopParent->addInitChild(*Node);
      else
        LoopParent->addChild(*Node);
    }
  } else {
    Trees[K.getID()].push_back(Node);
  }
}

static struct {
  bool operator()(KermaNode *A, KermaNode *B) {
    if (A->getRange().getStart() == B->getRange().getStart())
      return A->getRange().getEnd() > B->getRange().getEnd();
    return A->getRange().getStart() < B->getRange().getStart();
  }
} StartLocComparator;

// Create Nodes for each if statement. This function does not
// link nodes with their parents. We do that later.
void MATBuilder::CreateIfNodes(Kernel &Kernel, std::vector<KermaNode *> &All,
                               MemoryAccessInfo &MAI) {
  auto IFSourceRanges = SI.getIfRangesInRange(Kernel.getSourceRange());

  for (auto &IfRange : IFSourceRanges) {
    SourceRange FullRange;
    FullRange.setStart(std::get<0>(IfRange).getStart());
    if (std::get<2>(IfRange))
      FullRange.setEnd(std::get<2>(IfRange).getEnd());
    else
      FullRange.setEnd(std::get<1>(IfRange).getEnd());

    If *IfNode = new If(FullRange);
    IfNode->setCondRange(std::get<0>(IfRange));
    IfNode->setThenRange(std::get<1>(IfRange));
    IfNode->setElseRange(std::get<2>(IfRange));

    // Insert the BranchInsts corresponding to
    // the condition of this if node
    for (auto &BB : *Kernel.getFunction()) {
      for (auto &I : BB)
        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->getDebugLoc()) {
            SourceLoc BILoc(BI->getDebugLoc());
            if (IfNode->getCondRange().contains(BILoc) ||
                IfNode->getCondRange().containsLine(BILoc))
              IfNode->addConditionValue(BI);
          }
        }
    }

    IFNodes[Kernel.getID()].push_back(IfNode);
    All.push_back(IfNode);
  }
}

void MATBuilder::CreateLoopNode(Kernel &K, std::vector<KermaNode *> &All,
                                Loop &L) {
  auto *LN = new LoopNest(&L);
  LoopNodes[K.getID()].push_back(LN);
  All.push_back(LN);
  for (auto *SubLoop : L.getSubLoops()) {
    CreateLoopNode(K, All, *SubLoop);
  }
}

// Create Nodes for each for statement. This function does not
// link nodes with their parents. We do that later.
void MATBuilder::CreateLoopNodes(Kernel &Kernel, std::vector<KermaNode *> &All,
                                 LoopInfo &LI) {
  for (auto &L : LI)
    CreateLoopNode(Kernel, All, *L);
}

void MATBuilder::Nest(KermaNode *Node, KermaNode *Scope, Kernel &Kernel,
                      std::vector<KermaNode *> &All) {
  InsertNodeToParent(Node, Scope, Kernel);
  if ( isa<If>(Node) || isa<LoopNest>(Node)) {
    while (!All.empty() &&
        Node->getRange().contains((*All.begin())->getRange())) {
      auto *N = *All.begin();
      All.erase(All.begin());
      Nest(N, Node, Kernel, All);
    }
  }
}

bool MATBuilder::runOnModule(llvm::Module &M) {
  auto &Kernels = KI.getKernels();
  auto &MAI = getAnalysis<DetectMemoryAccessesPass>().getMemoryAccessInfo();

  for (auto &Kernel : Kernels) {
    if (MAI.getNumAccessesForKernel(Kernel) == 0)
      continue;

    auto &LI =
        getAnalysis<LoopInfoWrapperPass>(*Kernel.getFunction()).getLoopInfo();

    // Store all nodes in start loc order
    std::vector<KermaNode *> All;
    for (auto &E : MAI.getStmtsForKernel(Kernel)) {
      All.push_back(&E);
    }

    CreateIfNodes(Kernel, All, MAI);
    CreateLoopNodes(Kernel, All, LI);
    sort(All, StartLocComparator);

    while (!All.empty()) {
      auto *N = *All.begin();
      All.erase(All.begin());
      Nest(N, nullptr, Kernel, All);
    }
  }

  return false;
}

} // namespace kerma