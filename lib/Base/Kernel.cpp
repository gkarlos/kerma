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
  ArgAssume.resize(F->arg_size());
}

Kernel::Kernel(const Kernel& Other) {
  *this = Other;
}

namespace kerma {


// static hasAtLeastOneAssumption(K)

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Kernel &K) {
  OS << "(KERNEL) " << K.Range << " #" << K.getID() << " '" << K.getName() << '\'';
  if ( K.getLaunchAssumption())
    OS << ' ' << K.getLaunchAssumption()->getGrid() << ',' << K.getLaunchAssumption()->getBlock() << '\n';
  else
    OS << " <>,<>\n";
  OS << "         { ";
  if ( K.getFunction()->arg_size()) {
    if (K.ArgAssume[0])
      OS << "0:" << *K.ArgAssume[0];
    else
      OS <<  "0:-";
  }
  for ( size_t i = 1 ; i < K.getFunction()->arg_size(); ++i) {
    OS << ", " << i << ':';
    if ( auto *A = K.ArgAssume[i]) {
      OS << *A;
    } else {
      OS << '-';
    }
  }
  OS << " }";
  return OS;
}

}

