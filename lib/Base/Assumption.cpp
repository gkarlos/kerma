#include "kerma/Base/Assumption.h"


namespace kerma{

using namespace llvm;

bool DimAssumption::classof(const Assumption *A) {
  return A && A->getKind() == Assumption::AK_DIM;
}

bool LaunchAssumption::classof(const Assumption *A) {
  return A && A->getKind() == Assumption::AK_LAUNCH;
}

bool ValAssumption::classof(const Assumption *A) {
  return A && A->getKind() > Assumption::AK_DIM;
}

bool IAssumption::classof(const Assumption *A) {
  return A && A->getKind() == Assumption::AK_IVAL;
}

bool FPAssumption::classof(const Assumption *A) {
  return A && A->getKind() == Assumption::AK_FPVAL;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Assumption &A) {
  A.print(OS);
  return OS;
}

} // namespace kerma