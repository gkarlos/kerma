#include "kerma/Base/Memory.h"
#include <mutex>


using namespace kerma;

static std::mutex mtx;

static unsigned int genID() {
  static volatile unsigned int IDs = 0;
  unsigned int id;
  mtx.lock();
  id = IDs++;
  mtx.unlock();
  return id;
}

Memory::Memory(const std::string& Name, nvvm::AddressSpace::Ty AddrSpace, const Dim& D, bool DimAssumed)
: Memory(genID(), Name, AddrSpace, D, DimAssumed) {}

