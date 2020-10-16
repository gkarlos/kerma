#include "kerma/RT/Util.h"

#include "kerma/Support/Config.h"
#include "kerma/Transforms/Instrument/LinkDeviceRT.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"

#include <llvm/Support/raw_ostream.h>
#include <stdexcept>

using namespace llvm;

namespace {
  cl::OptionCategory RTLinkOptions("rt-link");
  cl::opt<std::string> OptInput(cl::Positional, cl::Required, cl::cat(RTLinkOptions), cl::desc("Input IR file"));
  cl::opt<bool> OptDump("dump", cl::cat(RTLinkOptions), cl::desc("Dump the resulting IR to stdout"), cl::init(false));
}

static std::string KermaDeviceRT = std::string(KERMA_HOME) + "/lib/RT/libKermaDeviceRT.bc";
static std::string CuMemtraceDeviceRT = std::string(KERMA_HOME) + "/lib/RT/libKermaDeviceRTCuMemtrace.bc";

int main(int argc, const char** argv) {
  llvm::cl::HideUnrelatedOptions(RTLinkOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  LLVMContext Context;
  SMDiagnostic Err;

  llvm::errs() << "Input: " << OptInput.getValue() << '\n';

  auto M = llvm::parseIRFile(OptInput.getValue(), Err, Context);

  if ( !kerma::KermaRTLinked(*M)) {
    llvm::errs() << "KermaRT is not linked with " << OptInput.getValue() << "\nLinking...";

    kerma::LinkDeviceRTPass LinkRTPass;
    LinkRTPass.useDeviceRT(KermaDeviceRT);

    try {
      LinkRTPass.runOnModule(*M);
    } catch ( std::runtime_error &e) {
      llvm::errs() << '\n' << e.what() << '\n';
      return 1;
    }
    llvm::errs() << (kerma::KermaRTLinked(*M)? " Success " : " Error ") << '\n';
  } else {
    llvm::errs() << "KermaRT is already linked\n";
  }

  if ( OptDump.getValue())
    M->print(llvm::outs(), nullptr);

  return 0;
}