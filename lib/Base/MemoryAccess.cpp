#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/Node.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/Support/raw_ostream.h>
#include <mutex>

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

MemoryAccess::MemoryAccess(Memory &M, llvm::Instruction *Inst, llvm::Value *Ptr,
                           MemoryAccess::Type Ty)
    : MemoryAccess(genID(), M, Inst, Ptr, Ty) {}

MemoryAccess::MemoryAccess(unsigned int ID, Memory &M, llvm::Instruction *Inst,
                           llvm::Value *Ptr, MemoryAccess::Type Ty)
    : KermaNode(NK_MA, SourceRange::Unknown, nullptr), ID(ID), M(M), Inst(Inst), Ptr(Ptr), Ty(Ty), Idx(Index::Unknown),
      DataDep(false), Bytes(1) {}

bool MemoryAccess::isSinglePointerAccess() {
  return Ty == Load || Ty == Store || Ty == Atomic || Ty == Memset;
}

bool MemoryAccess::isMemOperation() {
  return Ty == Memcpy || Ty == Memset || Ty == Memmove;
}

void MemoryAccess::validate() {
  auto Dim = M.getDim();
  if (!Idx.isUnknown() && !Dim.hasIndex(Idx))
    throw std::runtime_error("Invalid index " + Idx.toString() + " for dim " +
                             Dim.toString());
}

static std::vector<std::string> TyStrings = {"U",  "L",  "S", "A",
                                             "MC", "MM", "MS"};

const std::string &MemoryAccess::getTypeString() const {
  return TyStrings[getType()];
}

MemoryAccess &MemoryAccess::operator=(const MemoryAccess &Other) {
  KermaNode::operator=(Other);
  ID = Other.ID;
  M = Other.M;
  Inst = Other.Inst;
  Ptr = Other.Ptr;
  Loc = Other.Loc;
  Ty = Other.Ty;
  Idx = Other.Idx;
  DataDep = Other.DataDep;
  TransDataDep = Other.TransDataDep;
  Bytes = Other.Bytes;
  return *this;
}

bool MemoryAccess::operator==(const MemoryAccess &O) const {
  return M == O.M && Inst == O.Inst && Ptr == O.Ptr && Loc == O.Loc &&
         Ty == O.Ty && Idx == O.Idx;
}

void MemoryAccess::print(llvm::raw_ostream &OS, bool v) const {
  OS << "(" << getTypeString() << ")" << (isDataDependent() ? " +" : " -")
     << (isTransitivelyDataDependent() ? "+ " : "- ") << getLoc() << " #"
     << getID();
  OS << " :" << getNumBytes() << " . ";
  OS << getMemory();
  if ( Parent)
    OS << " . parent: #" << Parent->getID();
  else
    OS << " . parent: unknown";
  if (v)
    OS << " . " << Inst;
}

void MemoryAccess::print(std::ostream &OS, bool v) const {
  OS << "(" << getTypeString() << ")" << (isDataDependent() ? " +" : " -")
     << (isTransitivelyDataDependent() ? "+ " : "- ") << getLoc() << " #"
     << getID();
  OS << " :" << getNumBytes() << " . ";
  OS << getMemory();
  if (v)
    OS << " . " << Inst;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MemoryAccess &MA) {
  MA.print(os);
  return os;
}

std::ostream &operator<<(std::ostream &os, const MemoryAccess &MA) {
  MA.print(os);
  return os;
}

} // namespace kerma