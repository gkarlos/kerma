#ifndef KERMA_UTILS_LLVM_IO_H
#define KERMA_UTILS_LLVM_IO_H

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <system_error>

namespace kerma {


void writeModuleToFile(llvm::Module &M, const std::string &Path) {
  std::error_code Err;
  llvm::raw_ostream O(Path, Err);
  if (Err)
    throw std::runtime_error("writing module to file: " + Err.message());
  M.print(O, nullptr);
}

}


#endif // KERMA_UTILS_LLVM_IO_H