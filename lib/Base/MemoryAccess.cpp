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

MemoryAccess:: MemoryAccess(const Memory& Memory, llvm::Value *Ptr, MemoryAccess::Type Ty)
: MemoryAccess(genID(), Memory, Ptr, Ty) {}