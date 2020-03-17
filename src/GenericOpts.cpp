#include <llvm/Support/CommandLine.h>

using namespace llvm;

namespace kerma
{
namespace cl
{

bool KermaDebugFlag;
llvm::cl::opt<bool, true> Debug("debug", 
          llvm::cl::desc("Enable debug output"), 
          llvm::cl::Hidden, 
          llvm::cl::location(KermaDebugFlag));

} // cl
} // kerma

