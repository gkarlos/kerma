#include "kerma/Support/SourceInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <exception>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"

int main(int argc, const char** argv) {

  try {

    llvm::LLVMContext ctx;
    llvm::SMDiagnostic err;
    auto ir = llvm::parseIRFile(std::string(argv[1]), err, ctx);

    kerma::SourceInfo SI(ir.get(), argv[2]);

    // llvm::errs() << SI.getDirectory() << "\n";
    // llvm::errs() << SI.getFilename() << "\n";
    SI.getFunctionRange("asd");

  } catch(const std::exception &e) {
    llvm::errs() << e.what();
  }
  
  return 0;
}