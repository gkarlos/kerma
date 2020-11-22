#ifndef KERMA_ANALYSIS_DETECT_MEMORY_ACCESSES
#define KERMA_ANALYSIS_DETECT_MEMORY_ACCESSES

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Kernel.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/Memory.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/MemoryStmt.h"
#include "kerma/Base/Node.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <unordered_map>

namespace kerma {

class MemoryAccessInfo {
  friend class DetectMemoryAccessesPass;

private:
  std::unordered_map<unsigned, std::vector<MemoryAccess>> L;
  std::unordered_map<unsigned, std::vector<MemoryAccess>> S;
  std::unordered_map<unsigned, std::vector<MemoryAccess>> A;
  std::unordered_map<unsigned, std::vector<MemoryAccess>> MM; // memmove
  std::unordered_map<unsigned, std::vector<MemoryAccess>> MC; // memcpy
  std::unordered_map<unsigned, std::vector<MemoryAccess>> MS; // memset

  // The memory accesses grouped in statements, per kernel
  std::unordered_map<unsigned, std::vector<MemoryStmt>> MAS;

  // The Loop nests, per kernel
  std::unordered_map<unsigned, std::vector<LoopNest *>> Loops;

  // The actual nodes lists, per kernel
  std::unordered_map<unsigned, std::vector<KermaNode *>> Nodes;

  // Ignored accesses. Keys are kernel ids
  // Values are vectors of <instruction,underlying-object> pairs
  //
  // We can determine the reason these acceses were ignored by checking
  // the underlying object (pair->second) of an access (pair):
  //   * If its null: we couldn't find the access' memory
  //   * If its non-null the access is to local memory
  // Kernel struct arguments are in general passed by val unless they get too
  // large, this however is not really expressed in the IR so for now we just
  // assume all struct arguments are small enough to be in local memory. This
  // should be fine for most cases
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::LoadInst *, llvm::Value *>>>
      IgnL;
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::StoreInst *, llvm::Value *>>>
      IgnS;
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::CallInst *, llvm::Value *>>>
      IgnA;
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::MemMoveInst *, llvm::Value *>>>
      IgnMM; // memmove
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::MemCpyInst *, llvm::Value *>>>
      IgnMC; // memcpy
  std::unordered_map<unsigned,
                     std::vector<std::pair<llvm::MemSetInst *, llvm::Value *>>>
      IgnMS; // memset

public:
  // Retrieve the nodes of the Memory Tree for a given kernel.
  // The nodes are stored in level-order (BFS).
  // There is a catch. If any functions are inlined we might find
  // nodes with source locations outside of the kernel
  const std::vector<KermaNode *> &getMemoryTree(const Kernel &Kernel) {
    return Nodes[Kernel.getID()];
  }

  std::vector<MemoryAccess> getAccessesForKernel(const Kernel &K) {
    return getAccessesForKernel(K.getID());
  }
  std::vector<MemoryAccess> getAccessesForKernel(unsigned int ID);

  void addMemoryStmtForKernel(const Kernel &K, const MemoryStmt &Stmt) {
    MAS[K.getID()].push_back(Stmt);
  }

  std::vector<MemoryStmt> &getMemoryStmtsForKernel(const Kernel &K) {
    return MAS[K.getID()];
  }

  std::vector<MemoryStmt> &getMemoryStmtsForKernel(unsigned int ID) {
    return MAS[ID];
  }

  std::vector<std::pair<llvm::Instruction *, llvm::Value *>>
  getIgnoredAccessesForKernel(const Kernel &K);

  unsigned int getNumIgnoredAccesses();
  unsigned int getNumIgnoredAccessesForKernel(const Kernel &K) {
    return IgnL[K.getID()].size() + IgnS[K.getID()].size() +
           IgnA[K.getID()].size() + IgnMM[K.getID()].size() +
           IgnMC[K.getID()].size() + IgnMS[K.getID()].size();
  }

  unsigned int getNumAccesses();
  unsigned int getNumAccessesForKernel(const Kernel &K) {
    return L[K.getID()].size() + S[K.getID()].size() + A[K.getID()].size() +
           MM[K.getID()].size() + MC[K.getID()].size() + MS[K.getID()].size();
  }

  unsigned int getNumMemoryStmts();
  unsigned int getNumMemoryStmtsForKernel(const Kernel &K) {
    return MAS[K.getID()].size();
  }

  unsigned int getNumMemoryStmtsForKernel(unsigned int ID) {
    return MAS[ID].size();
  }

  unsigned int getNumLoadsForKernel(Kernel &K) { return L[K.getID()].size(); }
  const std::vector<MemoryAccess> &getLoadsForKernel(const Kernel &K) {
    return L[K.getID()];
  }

  unsigned int getNumStoresForKernel(Kernel &K) { return S[K.getID()].size(); }
  const std::vector<MemoryAccess> &getStoresForKernel(const Kernel &K) {
    return S[K.getID()];
  }

  unsigned int getNumAtomicsForKernel(Kernel &K) { return A[K.getID()].size(); }
  const std::vector<MemoryAccess> &getAtomicsForKernel(const Kernel &K) {
    return A[K.getID()];
  }

  unsigned int getNumMemmovesForKernel(Kernel &K) {
    return MM[K.getID()].size();
  }
  const std::vector<MemoryAccess> &getMemmovesForKernel(const Kernel &K) {
    return MM[K.getID()];
  }

  unsigned int getNumMemcpysForKernel(Kernel &K) {
    return MC[K.getID()].size();
  }
  const std::vector<MemoryAccess> &getMemcpysForKernel(const Kernel &K) {
    return MC[K.getID()];
  }

  unsigned int getNumMemsetsForKernel(Kernel &K) {
    return MS[K.getID()].size();
  }
  const std::vector<MemoryAccess> &getMemsetsForKernel(const Kernel &K) {
    return MS[K.getID()];
  }

  MemoryStmt *getMemoryStmtForAccess(const MemoryAccess &MA);
  MemoryStmt *getMemoryStmtAtRange(const SourceRange &R, bool strict = false);

  // Retrieve an access by ID
  MemoryAccess *getByID(unsigned int);

  // Check if an access is on the ignore list
  bool isIgnored(MemoryAccess &MA);
  bool isIgnored(unsigned int ID);
};

class DetectMemoryAccessesPass : public llvm::ModulePass {
public:
#ifdef KERMA_OPT_PLUGIN
  DetectMemoryAccessesPass();
#endif

public:
  static char ID;
  DetectMemoryAccessesPass(KernelInfo &KI, MemoryInfo &MI, SourceInfo &SI);
  bool runOnModule(llvm::Module &M) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  MemoryAccessInfo &getMemoryAccessInfo() { return MAI; }

  llvm::StringRef getPassName() const override {
    return "DetectMemoryAccessesPass";
  }

private:
  KernelInfo &KI;
  MemoryInfo &MI;
  SourceInfo &SI;
  MemoryAccessInfo MAI;

  void buildMemoryAccessInfo();
};

} // namespace kerma

#endif // KERMA_ANALYSIS_DETECT_MEMORY_ACCESSES