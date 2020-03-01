#include "kerma/Cuda/NVVM.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <kerma/passes/detect-addr-space/DetectAddrSpace.h>
#include <kerma/passes/detect-kernels/DetectKernels.h>

using namespace llvm;


/// Register Pass
char kerma::DetectAddrSpacePass::ID = 1;
static RegisterPass<kerma::DetectAddrSpacePass> INIT_DETECT_ADDR_SPACE("kerma-detect-addr-space", "Detect Address Space of Loads and Stores", false, true);


namespace kerma
{

void DetectAddrSpacePass::getAnalysisUsage(AnalysisUsage& AU) const 
{
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
}

bool DetectAddrSpacePass::runOnModule(llvm::Module &M) 
{
  auto kernels = &getAnalysis<DetectKernelsPass>().getKernels();

  for ( auto* kernel : *kernels) {
    outs() << "Detect-Address-Space: " << kernel->getFn()->getName() << "\n";
  }

  return false;
}

AddressSpace DetectAddrSpacePass::getAddrSpace(llvm::Value *v)
{
  if ( auto load = dyn_cast<LoadInst>(v)) {
    return getAddrSpace(load->getPointerOperand());
  }
}

}




