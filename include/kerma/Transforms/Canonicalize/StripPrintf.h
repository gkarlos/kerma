#ifndef KERMA_TRANSFORMS_STRIP_PRINTF_H
#define KERMA_TRANSFORMS_STRIP_PRINTF_H

#include "kerma/Analysis/DetectKernels.h"
#include <llvm-10/llvm/ADT/StringRef.h>
#include <llvm/Pass.h>

namespace kerma {


class StripPrintfPass : public llvm::ModulePass {
private:
  KernelInfo &KI;
public:
  static char ID;
  StripPrintfPass(KernelInfo &KI);
  bool runOnModule(llvm::Module& M) override;
  llvm::StringRef getPassName() { return "StripPrintfPass"; }
};

}

#endif // KERMA_TRANSFORMS_STRIP_PRINTF_H