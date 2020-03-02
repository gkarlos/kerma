#include <kerma/Support/LLVMFunctionShorthands.h>

namespace kerma
{

int getFnNumArgs(llvm::Function &fn)
{
  return fn.arg_size();
}

}