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

Kernel::Kernel(Function *F) : Kernel(genID(), F) {}

Kernel::Kernel(unsigned int ID, llvm::Function *F) : ID(ID), F(F) {
  assert(F);
  DemangledName = demangleFnWithoutArgs(*F);
}

Kernel::Kernel(const Kernel& Other) {
  *this = Other;
}

namespace kerma {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Kernel &K) {
  OS << "<KERNEL> " << K.Range << " #" << K.getID() << " '" << K.getName() << '\'';
  return OS;
}

}

