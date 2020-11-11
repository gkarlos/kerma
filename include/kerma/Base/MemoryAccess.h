#ifndef KERMA_BASE_MEMORY_ACCESS
#define KERMA_BASE_MEMORY_ACCESS

#include "kerma/Base/Memory.h"
#include "kerma/SourceInfo/SourceLoc.h"

namespace kerma {

class MemoryAccess {

public:
  enum Type : unsigned {
    Load,
    Store,
    Atomic,
    Cpy
  };

  MemoryAccess(unsigned int ID, Memory& M, llvm::Value *Ptr, MemoryAccess::Type Ty);
  MemoryAccess(Memory &M, llvm::Value *Ptr, MemoryAccess::Type Ty);

  unsigned int getID() { return ID; }

  const Memory& getMemory() const { return M; }
  MemoryAccess& setMemory(const Memory& Memory) {
    M = Memory;
    return *this;
  }

  const Type getType() const { return Ty; }
  MemoryAccess& setType(MemoryAccess::Type Ty) {
    this->Ty = Ty;
    return *this;
  }

  const llvm::Value *getPtr() { return Ptr; }
  MemoryAccess& setPtr(llvm::Value *Ptr) {
    this->Ptr = Ptr;
    return *this;
  }

  const SourceLoc &getLoc ()const  { return Loc; }

  MemoryAccess &setLoc(const SourceLoc &Loc) {
    this->Loc = Loc;
    return *this;
  }

private:
  unsigned int ID;
  Memory &M;
  llvm::Value *Ptr;
  SourceLoc Loc;
  MemoryAccess::Type Ty;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_ACCESS
