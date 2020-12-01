#include "kerma/Analysis/MemoryAccessTree.h"
#include "kerma/Base/MemoryAccess.h"

namespace kerma {

using namespace llvm;

void MemoryAccessTree::dump() {
  errs() << Kernel << '\n';
  for ( auto *Node : Tree)
    errs() << *Node << '\n';
  errs() << '\n';
}

void MemoryAccessTree::print(raw_ostream &OS) const {
  OS << Kernel << '\n';
  for ( auto *Node : Tree)
    OS << *Node << '\n';
  OS << '\n';
}

MemoryAccess *MemoryAccessTree::getAccessForInst(const llvm::Instruction *I) {
  return MAI->getForInst(I);
}

std::vector<MemoryAccess*> MemoryAccessTree::getAllAccesses() {
  return MAI->getAccessesForKernel(Kernel);
}

} // namespace kerma
