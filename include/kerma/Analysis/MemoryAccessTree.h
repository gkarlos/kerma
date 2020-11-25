#ifndef KERMA_ANALYSIS_MAT_H
#define KERMA_ANALYSIS_MAT_H

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Analysis/DetectMemoryAccesses.h"
#include "kerma/Base/If.h"
#include "kerma/Base/Kernel.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/Node.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Pass.h>
#include <memory>
#include <unordered_map>

namespace kerma {

class MemoryAccessTree {
public:
  MemoryAccessTree() = delete;
  friend class MATBuilder;

private:
  MemoryAccessTree(Kernel &Kernel, std::vector<KermaNode *> &Tree,
                   std::vector<LoopNest *> &LoopNodes,
                   std::vector<If *> &IfNodes)
      : Kernel(Kernel), Tree(Tree), LoopNodes(LoopNodes), IfNodes(IfNodes) {}

public:
  void dump();
  void print(llvm::raw_ostream &OS) const;

private:
  Kernel &Kernel;
  std::vector<KermaNode *> &Tree;
  std::vector<LoopNest *> &LoopNodes;
  std::vector<If *> &IfNodes;
};

// Pass

class MATBuilder : public llvm::ModulePass {
public:
  MATBuilder(KernelInfo &KI, MemoryInfo &MI, SourceInfo &SI);

  bool runOnModule(llvm::Module &M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  llvm::StringRef getPassName() const override { return "MATBuilder"; }

  // Get the tree for a kernel
  std::unique_ptr<MemoryAccessTree> get(Kernel &Kernel) {
    if (Trees.find(Kernel.getID()) == Trees.end())
      return nullptr;
    return std::unique_ptr<MemoryAccessTree>(new MemoryAccessTree(
        Kernel, Trees[Kernel.getID()], LoopNodes[Kernel.getID()],
        IFNodes[Kernel.getID()]));
  }

private:
  static char ID;
  KernelInfo &KI;
  MemoryInfo &MI;
  SourceInfo &SI;

  std::unordered_map<unsigned, std::vector<LoopNest *>> LoopNodes;
  std::unordered_map<unsigned, std::vector<If *>> IFNodes;
  std::unordered_map<unsigned, std::vector<KermaNode *>> Trees;

  void CreateIfNodes(Kernel &Kernel, std::vector<KermaNode *> &All,
                     MemoryAccessInfo &MAI);
  void CreateLoopNode(Kernel &Kernel, std::vector<KermaNode *> &All,
                      llvm::Loop &L);
  void CreateLoopNodes(Kernel &Kernel, std::vector<KermaNode *> &All,
                       llvm::LoopInfo &LI);
  void PopulateLoop(Kernel &K, LoopNest *LN, std::vector<KermaNode *> &All);
  void PopulateIf(Kernel &K, If *IFnode, std::vector<KermaNode *> &All);
  void Nest(KermaNode *N, KermaNode *Scope, Kernel &Kernel,
            std::vector<KermaNode *> &All);

  void InsertNodeToParent(KermaNode *Node, KermaNode *Parent, Kernel &K);
};

} // namespace kerma

#endif // KERMA_ANALYSIS_MAT_H
