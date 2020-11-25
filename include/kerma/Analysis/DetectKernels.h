#ifndef KERMA_PASS_DETECT_KERNELS_H
#define KERMA_PASS_DETECT_KERNELS_H

#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <vector>

#include "kerma/Base/Kernel.h"

namespace kerma {

class KernelInfo {
private:
  std::vector<Kernel> Kernels;

public:
  KernelInfo()=delete;
  // KernelInfo(const std::vector<Kernel> &Kernels) : Kernels(Kernels) {}
  KernelInfo(std::vector<Kernel> &Kernels) : Kernels(Kernels) {}
  bool isKernel(llvm::Function &F);
  Kernel *findByID(unsigned ID);
  Kernel *find(llvm::Function *F);
  std::vector<Kernel> &getKernels() { return Kernels; }
  KernelInfo& operator=(const KernelInfo &O) {
    Kernels = O.Kernels;
    return *this;
  }
  Kernel *getKernelForFunction(llvm::Function &F) {
    for ( auto &Kernel : Kernels)
      if ( Kernel.getFunction() == &F)
        return &Kernel;
    return nullptr;
  }
};

class DetectKernelsPass : public llvm::ModulePass {
public:
  static char ID;
  DetectKernelsPass();

public:
  bool runOnModule(llvm::Module &M) override;
  virtual void print(llvm::raw_ostream &O, const llvm::Module *M) const override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

public:
  /// Get a vector containing all the kernel functions
  /// found in the module. The returned vector is a copy
  /// the internal vector of the pass and can be freely
  /// manipulated
  const std::vector<Kernel> & getKernels();
  void getKernels(std::vector<Kernel>& Kernels);
  llvm::StringRef getPassName() const override { return "DetectKernelsPass"; }
private:
  std::vector<Kernel> Kernels;
};

std::unique_ptr<DetectKernelsPass> createDetectKernelsPass();


/// Extract the kernels from a module. Results are cached
/// until:
///   (a) invalidateCacheEntry is true in which case the
///       cache entry for this module is recomputed.
///   (b) clearCache() is called, in which case the entire
///       cache is invalided but not recomputed.
std::vector<llvm::Function*> getKernelFunctions( const llvm::Module &M, bool invalidateCacheEntry=false);
const std::vector<Kernel> & getKernels(const llvm::Module &M, bool invalidateCacheEntry=false);

/// Check if a function is a CUDA kernel
bool isKerneFunction(const llvm::Function& F);

} // end namespace kerma

#endif