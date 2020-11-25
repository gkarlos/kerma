#ifndef KERMA_BASE_MEMORY_H
#define KERMA_BASE_MEMORY_H

#include "kerma/Base/Dim.h"
// #include "kerma/Base/Kernel.h"
#include "kerma/NVVM/NVVM.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>

#include <set>

namespace kerma {

/// Represents the various memories used by a kernel
class Memory {
public:
  enum Kind : unsigned { Unknown = 0, Arg, Global, Alloca};

public:
  Memory(const std::string &Name,
         nvvm::AddressSpace::Ty AddrSpace = nvvm::AddressSpace::Unknown);
  Memory(unsigned ID, const std::string &Name,
         nvvm::AddressSpace::Ty AddrSpace = nvvm::AddressSpace::Unknown);
  Memory(const std::string &Name, nvvm::AddressSpace::Ty AddrSpace,
         const Dim &KnownDim, const Dim &AssumedDim, llvm::Value *V);
  Memory(unsigned ID, const std::string &Name, nvvm::AddressSpace::Ty AddrSpace,
         const Dim &KnownDim, const Dim &AssumedDim, llvm::Value *V);

  unsigned getID() const { return this->ID; }

  const std::string getName() const { return Name; }
  Memory &setName(const std::string &Name);
  Memory &setName(const char *Name);
  Memory &setName(const llvm::StringRef &Name);

  llvm::Value *getValue() const { return V; }
  llvm::Type *getType() const { return T; }

  unsigned getTypeSize() const { return TySize; }

  Memory &setValue(llvm::Value *V);
  Memory &setType(llvm::Type *T);
  Memory &setTypeSize(unsigned SZ);

  // Memory &addKernelUser(const Kernel& Kernel);
  Memory &addKernelUser(unsigned int KernelID);
  // Memory &removeKernelUser(const Kernel& Kernel);
  Memory &removeKernelUser(unsigned int KernelID);

  // bool hasKernelUser(const Kernel& Kernel);
  bool hasKernelUser(unsigned int KernelID);


  const Dim& getKnownDim() const { return KnownDim; }
  const Dim& getAssumedDim() const { return AssumedDim; }
  const Dim& getDim() const { return AssumedDim? AssumedDim : KnownDim; }
  Memory &setKnownDim(const Dim& Dim);
  Memory &setAssumedDim(const Dim& Dim);
  Memory &assumeDim(const Dim &Dim);

  const nvvm::AddressSpace::Ty &getAddrSpace() const { return AddrSpace; }

  bool isArgument() const { return Kind == Arg; }
  bool isGlobal() const { return Kind == Global; }
  bool isAlloca() const { return Kind == Alloca; }

  enum Kind getKind() const { return Kind; }
  Memory &setKind(enum Kind Kind) {
    this->Kind = Kind;
    return *this;
  }

  bool operator==(const Memory &Other) { return ID == Other.ID; }
  bool operator<(const Memory &Other) { return ID < Other.ID; }
  Memory &operator=(const Memory &Other) {
    ID = Other.ID;
    Kind = Other.Kind;
    Name = Other.Name;
    AssumedDim = Other.AssumedDim;
    KnownDim = Other.KnownDim;
    AddrSpace = Other.AddrSpace;
    V = Other.V;
    return *this;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Memory &M);

  friend std::ostream &operator<<(std::ostream &os, const Memory &M);

  friend void swap(Memory &A, Memory &B) {
    using std::swap;
    swap(A.ID, B.ID);
    swap(A.Kind, B.Kind);
    swap(A.Name, B.Name);
    swap(A.KnownDim, B.KnownDim);
    swap(A.AssumedDim, B.AssumedDim);
    swap(A.AddrSpace, B.AddrSpace);
    swap(A.V, B.V);
    swap(A.T, B.T);
    swap(A.KernelUsers, B.KernelUsers);
    swap(A.TySize, B.TySize);
  }

  private:
    unsigned int ID;
    std::string Name;
    Kind Kind;
    Dim KnownDim;
    Dim AssumedDim;
    nvvm::AddressSpace::Ty AddrSpace;
    llvm::Value *V = nullptr;
    llvm::Type *T = nullptr;
    std::set<unsigned> KernelUsers;
    unsigned TySize = 0;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_H