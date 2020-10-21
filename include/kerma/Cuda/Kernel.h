#ifndef KERMA_CUDA_KERNEL_H
#define KERMA_CUDA_KERNEL_H

#include <llvm/IR/Function.h>

namespace kerma {

class Kernel {
private:
  llvm::Function *F;
  std::string DemangledName;
  unsigned int ID;

public:
  explicit Kernel(llvm::Function *F);
  Kernel(llvm::Function *F, unsigned int ID);
  Kernel(const Kernel& Other);
  Kernel& operator=(const Kernel& Other);
  bool operator==(const Kernel& Kernel);
  bool operator==(const llvm::Function& F);

  llvm::Function *getFunction() const;
  std::string getDemangledName() const;
  unsigned int getID() const;
};

} // namespace kerma



#endif