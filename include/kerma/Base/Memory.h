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
  enum Type: unsigned {Arg, Global, Alloca, Unknown};

private:
  unsigned int ID;
  Type Ty;
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

  void setName(const std::string& Name) { this->Name = Name; }
  void setName(const char *Name) { this->Name = Name? Name : ""; }
  void setName(const llvm::StringRef& Name) { this->Name = Name.str(); }

  const std::string& getName() { return Name; }

  void setDim(const Dim& Dim) { D = Dim; }
  void setDim(unsigned int x, unsigned int y=1, unsigned int z=3) { D = Dim(x,y,z); }

  const Dim& getDim() { return D; }

  bool dimIsAssumed() { return DimAssumed; }

  const nvvm::AddressSpace::Ty& getAddrSpace() { return AddrSpace; }

  void setValue(llvm::Value *V) { this->V = V; }
  const llvm::Value *getValue() const { return V; }
  bool hasValue() const { return V != nullptr; }

  bool isArgument() { return Ty == Arg; }
  bool isGlobal() { return Ty == Global; }
  bool isAlloca() { return Ty = Alloca; }

  bool operator==(const Memory& Other) { return ID = Other.ID; }
  bool operator<(const Memory& Other) { return ID < Other.ID; }

  enum Memory::Type getType() { return Ty; }
};

}


#endif // KERMA_BASE_MEMORY_H