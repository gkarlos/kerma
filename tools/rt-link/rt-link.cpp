#include "kerma/RT/Util.h"

#include "kerma/Transforms/LinkKermaRTDevice.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <stdexcept>

using namespace llvm;

namespace {
  cl::OptionCategory RTLinkOptions("rt-link");
  cl::opt<std::string> OptInput(cl::Positional, cl::Required, cl::cat(RTLinkOptions), cl::desc("Input IR file"));
  cl::opt<bool> OptDump("dump", cl::cat(RTLinkOptions), cl::desc("Dump the resulting IR to stdout"), cl::init(false));
}

int main(int argc, const char** argv) {
  llvm::cl::HideUnrelatedOptions(RTLinkOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  LLVMContext Context;
  SMDiagnostic Err;

  auto M = llvm::parseIRFile(argv[1], Err, Context);

  if ( !kerma::KermaRTLinked(*M)) {
    std::cout << "KermaRT is not linked with " << argv[1] << '\n';
    std::cout << "Linking...";
    kerma::LinkKermaRTDevicePass LinkRTPass;
    try {
      LinkRTPass.runOnModule(*M);
    } catch ( std::runtime_error &e) {
      std::cout << '\n' << e.what() << '\n';
      return 1;
    }
    std::cout << (kerma::KermaRTLinked(*M)? " Success " : " Error ") << '\n';
  } else {
    std::cout << "KermaRT is already linked\n";
  }

  if ( OptDump.getValue())
    M->print(llvm::errs(), nullptr);

  return 0;
}