#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace kerma {

class MaterializeBlockThreadPass : public llvm::FunctionPass {

public:
  static char ID;
  MaterializeBlockThreadPass();

public:
  bool runOnFunction(llvm::Function &F) override;
};

} // namespace kerma