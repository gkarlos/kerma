#ifndef KERMA_ANALYSIS_DETECT_MEMORIES_H
#define KERMA_ANALYSIS_DETECT_MEMORIES_H

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Base/Kernel.h"
#include "kerma/Base/Memory.h"
#include "kerma/NVVM/NVVM.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Pass.h>
#include <unordered_map>

namespace kerma {

using MemMap = std::unordered_map<llvm::Value *, Memory>;

class MemoryInfo {
  friend class DetectMemoriesPass;
private:
  // std::unordered_map<unsigned, MemMap> Memories;
  std::unordered_map<unsigned, std::vector<Memory>> M;

public:
  MemoryInfo()=default;
  const std::vector<Memory>& getForKernel(const Kernel &Kernel) {
    return getForKernel(Kernel.getID());
  }
  const std::vector<Memory>& getForKernel(unsigned KernelID);

  std::vector<Memory> getArgMemoriesForKernel(const Kernel &Kernel) {
    return getArgMemoriesForKernel(Kernel.getID());
  }
  std::vector<Memory> getArgMemoriesForKernel(unsigned KernelID);

  Memory *getMemoryForArg(llvm::Argument *Arg, Kernel *K=nullptr);
  Memory *getMemoryForVal(llvm::Value *V, Kernel *K=nullptr);

  unsigned getArgMemCount();
  unsigned getGVMemCount();

};

class DetectMemoriesPass : public llvm::ModulePass {
public:
#ifdef KERMA_OPT_PLUGIN
  DetectMemoriesPass();
#endif

  DetectMemoriesPass(KernelInfo &KI, bool SkipLocal = true);

private:
  KernelInfo &KI;
  bool SkipLocal;
  MemoryInfo MI;

public:
  static char ID;
  bool runOnModule(llvm::Module &M) override;
  const MemoryInfo& getMemoryInfo() { return MI; }
};



} // namespace kerma

#endif // KERMA_ANALYSIS_DETECT_MEMORIES_H