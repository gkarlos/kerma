#include "kerma/Base/Memory.h"
#include "kerma/Base/Index.h"
#include "kerma/NVVM/NVVM.h"
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <mutex>
#include <atomic>

namespace kerma {

using namespace llvm;

static std::mutex mtx;

static unsigned int genID() {
  static volatile unsigned int IDs = 0;
  unsigned int id;
  mtx.lock();
  id = IDs++;
  mtx.unlock();
  return id;
}

Memory::Memory(const std::string &Name, nvvm::AddressSpace::Ty AddrSpace)
    : Memory(genID(), Name, AddrSpace) {}

Memory::Memory(unsigned ID, const std::string &Name,
               nvvm::AddressSpace::Ty AddrSpace)
    : Memory(ID, Name, AddrSpace, Dim::None, Dim::None, nullptr) {}

Memory::Memory(const std::string &Name, nvvm::AddressSpace::Ty AddrSpace,
               const Dim &KnownDim, const Dim &AssumedDim, llvm::Value *V)
    : Memory(genID(), Name, AddrSpace, KnownDim, AssumedDim, V) {}

Memory::Memory(unsigned ID, const std::string &Name,
               nvvm::AddressSpace::Ty AddrSpace, const Dim &KnownDim,
               const Dim &AssumedDim, llvm::Value *V)
    : ID(ID), Name(Name), AddrSpace(AddrSpace), KnownDim(KnownDim),
      AssumedDim(AssumedDim), V(V) {}

Memory &Memory::setName(const std::string &Name) {
  this->Name = Name;
  return *this;
}

Memory &Memory::setName(const char *Name) {
  this->Name = Name ? Name : "";
  return *this;
}

Memory &Memory::setName(const llvm::StringRef &Name) {
  this->Name = Name.str();
  return *this;
}

Memory &Memory::setKnownDim(const Dim &Dim) {
  this->KnownDim = Dim;
  return *this;
}

Memory &Memory::setAssumedDim(const Dim &Dim) {
  this->AssumedDim = Dim;
  return *this;
}

Memory &Memory::assumeDim(const Dim &Dim) { return setAssumedDim(Dim); }

Memory &Memory::setValue(llvm::Value *V) {
  this->V = V;
  return *this;
}

Memory &Memory::setType(llvm::Type *T) {
  this->T = T;
  if (T && this->V) {
    if (this->isArgument()) {
      TySize = cast<Argument>(V)
                   ->getParent()
                   ->getParent()
                   ->getDataLayout()
                   .getTypeAllocSize(T);
    } else if (this->isGlobal()) {
      TySize = cast<GlobalVariable>(V)
                   ->getParent()
                   ->getDataLayout()
                   .getTypeAllocSize(T);
    }
  }
  return *this;
}

Memory &Memory::setTypeSize(unsigned int SZ) {
  this->TySize = SZ;
  return *this;
}

Memory &Memory::addKernelUser(const Kernel& Kernel) {
  return addKernelUser(Kernel.getID());
}

Memory &Memory::addKernelUser(unsigned int KernelID) {
  this->KernelUsers.insert(KernelID);
  return *this;
}

Memory &Memory::removeKernelUser(const Kernel& Kernel) {
  return removeKernelUser(Kernel.getID());
}

Memory &Memory::removeKernelUser(unsigned int KernelID) {
  this->KernelUsers.erase(KernelID);
  return *this;
}

bool Memory::hasKernelUser(const Kernel& Kernel) {
  return hasKernelUser(Kernel.getID());
}

bool Memory::hasKernelUser(unsigned int KernelID) {
  return KernelUsers.find(KernelID) != KernelUsers.end();
}

static std::string KindToString(enum Memory::Kind K) {
  switch (K) {
  case Memory::Kind::Arg:
    return "a";
  case Memory::Kind::Global:
    return "g";
  case Memory::Kind::Alloca:
    return "l";
  default:
    return "u";
  };
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Memory &M) {
  OS << '\'' << M.getName() << "':" << M.TySize << " !" << M.getAddrSpace().getID() << ' ';
  OS << M.KnownDim;
  OS << ",";
  OS << M.AssumedDim;
  return OS;
}

std::ostream &operator<<(std::ostream &OS, const Memory &M) {
  OS << '\'' << M.getName() << "':" << M.TySize << " !" << M.getAddrSpace().getID() << ' ';
  OS << M.KnownDim;
  OS << ",";
  OS << M.AssumedDim;
  return OS;
}

} // namespace kerma
