//
// Created by gkarlos on 1/5/20.
//

#include <kerma/cuda/CudaKernel.h>
#include <kerma/Support/Demangle.h>

namespace kerma
{
namespace cuda
{

void CudaKernel::pp(llvm::raw_ostream& os) {

  std::string demangled = demangleFn(this->fn_);

  os << demangled
     << ((demangled != this->fn_->getName())? " (demangled)" : "") << "\n"
     << " " << u8"└" << " In " << cudaSideToString(this->irModuleSide_)
     << "-side module:" << this->fn_->getParent()->getName() << "\n";
}

}
}

