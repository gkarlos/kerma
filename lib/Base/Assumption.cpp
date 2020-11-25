#include "kerma/Base/Assumption.h"
#include <llvm-10/llvm/Support/Casting.h>


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

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Assumption &A) {
  A.print(OS);
  return OS;
}

// ValAssumption &ValAssumption::operator=(const ValAssumption &O) {
//   Assumption::operator=(O);
//   if ( auto *IA = dyn_cast<IAssumption>(&O))
//     *this = *IA;
//   else
//     *this = *(dyn_cast<FPAssumption>(&O));
//   return *this;
// }

} // namespace kerma