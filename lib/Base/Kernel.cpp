#include "kerma/Base/Kernel.h"
#include "kerma/Support/Demangle.h"
#include <llvm/Support/ManagedStatic.h>

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

Kernel::Kernel(Function *F) : Kernel(F, genID()) {}

Kernel::Kernel(Function *F, unsigned int ID) : F(F), ID(ID) {
  assert(F);
  DemangledName = demangleFnWithoutArgs(*F);
}

Kernel::Kernel(const Kernel& Other)
: F(Other.F), DemangledName(Other.DemangledName), ID(Other.ID) {}

Kernel& Kernel::operator=(const Kernel &Other) {
  F = Other.F;
  DemangledName = Other.DemangledName;
  ID = Other.ID;
  return *this;
}

bool Kernel::operator==(const Kernel &Other) { return ID == Other.ID; }

bool Kernel::operator==(const Function& F) { return this->F == &F; }

Function * Kernel::getFunction() const { return F; }

std::string Kernel::getDemangledName() const { return DemangledName; }

unsigned int Kernel::getID() const { return ID; }