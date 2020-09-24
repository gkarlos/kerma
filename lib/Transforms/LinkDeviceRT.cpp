#include "kerma/Transforms/LinkDeviceRT.h"

#include "kerma/RT/Util.h"
#include "kerma/Support/Config.h"

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace kerma {

char LinkDeviceRTPass::ID = 5;

static std::string KermaDeviceRT = std::string(KERMA_HOME) + "/lib/RT/libKermaDeviceRT.bc";

LinkDeviceRTPass::LinkDeviceRTPass() : LinkDeviceRTPass(KermaDeviceRT) {}
LinkDeviceRTPass::LinkDeviceRTPass(std::string RTPath) : DeviceRTPath(RTPath), ModulePass(ID) {}

void LinkDeviceRTPass::useDeviceRT(std::string RTPath) { DeviceRTPath = RTPath; }

bool LinkDeviceRTPass::runOnModule(llvm::Module &M) {
  if ( M.getTargetTriple().find("nvptx") == std::string::npos)
    return false;

  // std::string LibRT = std::string(KERMA_HOME) + "/lib/RT/libKermaRT.bc";

  SMDiagnostic Err;
  LLVMContext &Ctx = M.getContext();

  auto LibRTModule = llvm::parseIRFile(DeviceRTPath, Err, Ctx);
  if ( LibRTModule.get() == nullptr)
    throw KermaRTIRParseError(Err.getMessage());

  Linker::linkModules(M, std::move(LibRTModule));

  return true;
}

static RegisterPass<LinkDeviceRTPass> RegisterLinkDeviceRTPass(
        /* pass arg  */   "kerma-link-device-rt",
        /* pass name */   "Link the Kerma device RT",
        /* modifies CFG */ false,
        /* analysis pass*/ false);

} // namespace kerma