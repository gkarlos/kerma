#ifndef KERMA_BASE_MEMORY_H
#define KERMA_BASE_MEMORY_H

#include "kerma/Base/Dim.h"
#include "kerma/NVVM/NVVM.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>

namespace kerma {

class Memory {
public:
  enum Pos: unsigned { Arg, Global, Alloca, Unknown};

private:
  unsigned int ID;
  Pos Pos=Unknown;
  std::string Name;
  bool DimAssumed;
  Dim D;
  nvvm::AddressSpace::Ty AddrSpace;
  llvm::Value *V=nullptr;


public:
  Memory(unsigned int ID, const std::string& Name, nvvm::AddressSpace::Ty AddrSpace, const Dim& D, bool DimAssumed=false)
  : ID(ID), Name(Name), AddrSpace(AddrSpace), D(D), DimAssumed(DimAssumed) {}

  Memory(unsigned int ID, const std::string& Name, nvvm::AddressSpace::Ty AddrSpace)
  : Memory(ID, Name, AddrSpace, Dim::None) {}

  Memory(const std::string& Name, nvvm::AddressSpace::Ty AddrSpace)
  : Memory(Name, AddrSpace, Dim::None) {}

  Memory(const std::string& Name, nvvm::AddressSpace::Ty AddrSpace, const Dim& D, bool DimAssumed=false);

  unsigned int getID() const { return ID; }

  Memory& setName(const std::string& Name) {
    this->Name = Name;
    return *this;
  }

  Memory& setName(const char *Name) {
    this->Name = Name? Name : "";
    return *this;
  }

  Memory& setName(const llvm::StringRef& Name) {
    this->Name = Name.str();
    return *this;
  }

  const std::string& getName() const { return Name; }

  Memory& setDim(const Dim& Dim) {
    D = Dim;
    return *this;
  }

  Memory& setDim(unsigned int x, unsigned int y=1, unsigned int z=3) {
    D = Dim(x,y,z);
    return *this;
  }

  const Dim& getDim() const { return D; }

  bool dimIsAssumed() const { return DimAssumed; }

  const nvvm::AddressSpace::Ty& getAddrSpace() const { return AddrSpace; }

  Memory& setValue(llvm::Value *V) {
    this->V = V;
    return *this;
  }

  const llvm::Value *getValue() const { return V; }
  bool hasValue() const { return V != nullptr; }

  bool isArgument() const { return Pos == Arg; }
  bool isGlobal() const { return Pos == Global; }
  bool isAlloca() const { return Pos == Alloca; }

  enum Pos getPos() const { return Pos; }
  Memory& setPos(enum Pos Pos) {
    this->Pos = Pos;
    return *this;
  }

  bool operator==(const Memory& Other) { return ID = Other.ID; }
  bool operator<(const Memory& Other) { return ID < Other.ID; }
  Memory& operator=(const Memory& Other) {
    ID = Other.ID;
    Pos = Other.Pos;
    Name = Other.Name;
    DimAssumed = Other.DimAssumed;
    D = Other.D;
    AddrSpace = Other.AddrSpace;
    V = Other.V;
    return *this;
  }
};

}


#endif // KERMA_BASE_MEMORY_H