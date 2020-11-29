#ifndef KERMA_BASE_KERNEL_H
#define KERMA_BASE_KERNEL_H

// #include "kerma/Analysis/MemoryAccessTree.h"
#include "kerma/Base/Assumption.h"
#include "kerma/Base/Index.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/ADT/iterator.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Function.h>
#include <unordered_map>

namespace kerma {

class MemoryAccessTree;

class Kernel {
private:
  llvm::Function *F;
  std::string DemangledName;
  unsigned ID;
  SourceRange Range;
  LaunchAssumption *LaunchAssume=nullptr;
  std::vector<Assumption*> ArgAssume;
  bool hasArgAssumptions = false;
  MemoryAccessTree *MAT=nullptr;
  // unsigned WarpIndex=0;
  // Index BlockIndex=Index(0,0,0);
  // Index ThreadIndex=Index(0,0,0);

public:
  Kernel()=delete;
  explicit Kernel(llvm::Function *F);
  Kernel(unsigned int ID, llvm::Function *F);
  Kernel(const Kernel& Other);

  Kernel& operator=(const Kernel &Other) {
    F = Other.F;
    DemangledName = Other.DemangledName;
    ID = Other.ID;
    Range = Other.Range;
    LaunchAssume = Other.LaunchAssume;
    ArgAssume = Other.ArgAssume;
    return *this;
  }

  bool operator==(const Kernel &Other) { return ID == Other.ID; }
  bool operator==(const llvm::Function& F) { return this->F == &F; }

  // const Index &getBlock() { return BlockIndex; }
  // const Index &getThread() { return ThreadIndex; }
  // unsigned getWarp() { return WarpIndex; }

  void attachMAT(MemoryAccessTree *MAT) {
    this->MAT = MAT;
  }

  MemoryAccessTree *getMAT() { return this->MAT; }

  llvm::Function *getFunction() const { return F; }
  unsigned getID() const { return ID; }

  Kernel& setSourceRange(const SourceRange& Range) {
    this->Range = Range;
    return *this;
  }
  Kernel& setLaunchAssumption(LaunchAssumption *LA) {
    LaunchAssume = LA;
    return *this;
  }
  Kernel& setArgAssumption(unsigned argidx, Assumption &A) {
    if ( argidx >= ArgAssume.size())
      throw std::runtime_error("argidx out of range");
    ArgAssume[argidx] = &A;
    return *this;
  }

  const LaunchAssumption *getLaunchAssumption() const {
    return LaunchAssume;
  }

  SourceRange& getSourceRange() { return Range; }
  const std::string& getName() const { return DemangledName; }
  const std::string& getDemangledName() const { return DemangledName; };

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Kernel &K);
};

} // namespace kerma



#endif