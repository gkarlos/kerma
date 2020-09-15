#include "kerma/RT/Util.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>

using namespace llvm;

int main(int argc, const char** argv) {
  if ( argc < 2) {
    std::cout << "Usage: rt-check <ir_file>";
    return 0;
  }

  LLVMContext Context;
  SMDiagnostic Err;

  auto M = llvm::parseIRFile(argv[1], Err, Context);

  std::cout << "KermaRT is" << (kerma::rt::KermaRTLinked(*M)? " " : " NOT ") << "linked with " << argv[1] << '\n';

  return 0;
}