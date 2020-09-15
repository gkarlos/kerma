#include "kerma/Transforms/LinkKermaRTDevice.h"

#include "kerma/RT/Util.h"
#include "kerma/Support/Config.h"

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace kerma {

char LinkKermaRTDevicePass::ID = 5;

LinkKermaRTDevicePass::LinkKermaRTDevicePass() : ModulePass(ID) {}

bool LinkKermaRTDevicePass::runOnModule(llvm::Module &M) {
  if ( M.getTargetTriple().find("nvptx") == std::string::npos)
    return false;

  std::string LibRT = std::string(KERMA_HOME) + "/lib/RT/libKermaRT.bc";

  SMDiagnostic Err;
  LLVMContext &Ctx = M.getContext();

  auto LibRTModule = llvm::parseIRFile(LibRT, Err, Ctx);
  if ( LibRTModule.get() == nullptr)
    throw KermaRTIRParseError(Err.getMessage());

  Linker::linkModules(M, std::move(LibRTModule));

  return true;
}

static RegisterPass<LinkKermaRTDevicePass> RegisterLinkKermaRTDevicePass(
        /* pass arg  */   "kerma-link-rt-device",
        /* pass name */   "Link the Kerma device RT",
        /* modifies CFG */ false,
        /* analysis pass*/ false);

} // namespace kerma