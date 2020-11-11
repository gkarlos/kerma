#include "kerma/Base/MemoryAccess.h"

#include <mutex>

using namespace kerma;
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

MemoryAccess::MemoryAccess(Memory &M, llvm::Value *Ptr, MemoryAccess::Type Ty)
: MemoryAccess(genID(), M, Ptr, Ty) {}
MemoryAccess::MemoryAccess(unsigned int ID, Memory& M, llvm::Value *Ptr, MemoryAccess::Type Ty)
: ID(ID), M(M), Ptr(Ptr), Ty(Ty) {}

