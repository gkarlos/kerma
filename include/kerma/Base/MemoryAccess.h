#ifndef KERMA_BASE_MEMORY_ACCESS
#define KERMA_BASE_MEMORY_ACCESS

#include "kerma/Base/Index.h"
#include "kerma/Base/Memory.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include <llvm/IR/Instruction.h>
#include <map>
#include <ostream>

namespace kerma {

class MemoryAccess {
public:
  enum Type : unsigned {
    Uknown = 0,
    Load,
    Store,
    Atomic,
    Memcpy,
    Memmove,
    Memset,
  };

  MemoryAccess(Memory &M, llvm::Instruction *Inst, llvm::Value *Ptr,
               MemoryAccess::Type Ty);

  unsigned int getID() const { return ID; }

  const Memory &getMemory() const { return M; }
  MemoryAccess &setMemory(const Memory &Memory) {
    M = Memory;
    return *this;
  }

  const Type getType() const { return Ty; }
  const std::string &getTypeString() const;
  MemoryAccess &setType(MemoryAccess::Type Ty) {
    this->Ty = Ty;
    return *this;
  }

  const Index &getIndex() const { return Idx; }
  MemoryAccess &setIndex(const Index &Idx) {
    this->Idx = Idx;
    return *this;
  }

  const SourceLoc &getLoc() const { return Loc; }
  MemoryAccess &setLoc(const SourceLoc &Loc) {
    this->Loc = Loc;
    return *this;
  }

  llvm::Instruction *getInst() { return Inst; }
  MemoryAccess &setInst(llvm::Instruction *Inst) {
    this->Inst = Inst;
    return *this;
  }

  llvm::Value *getPtr() { return Ptr; }
  MemoryAccess &setPtr(llvm::Value *Ptr) {
    this->Ptr = Ptr;
    return *this;
  }

  bool isSinglePointerAccess();
  bool isMemOperation();
  bool isCallInst();

  void validate();

  MemoryAccess &operator=(const MemoryAccess &Other);

  bool operator==(const MemoryAccess &O) const;

  void print(llvm::raw_ostream &OS, bool v=false) const;
  void print(std::ostream &OS, bool v=false) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const MemoryAccess &MA);
  friend std::ostream &operator<<(std::ostream &os, const MemoryAccess &MA);

  friend void swap(MemoryAccess &A, MemoryAccess &B) {
    using std::swap;
    swap(A.ID, B.ID);
    swap(A.M, B.M);
    swap(A.Inst, B.Inst);
    swap(A.Ptr, B.Ptr);
    swap(A.Loc, B.Loc);
    swap(A.Ty, B.Ty);
    swap(A.Idx, B.Idx);
  }

private:
  MemoryAccess(unsigned int ID, Memory &M, llvm::Instruction *Inst,
               llvm::Value *Ptr, MemoryAccess::Type Ty);

private:
  unsigned int ID;
  Memory &M;
  llvm::Instruction *Inst;
  llvm::Value *Ptr;
  SourceLoc Loc;
  MemoryAccess::Type Ty;
  Index Idx;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_ACCESS
