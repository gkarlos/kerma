#ifndef KERMA_TRANSFORMS_LINK_KERMA_RT_DEVICE_PASS_H
#define KERMA_TRANSFORMS_LINK_KERMA_RT_DEVICE_PASS_H

#include "llvm/Pass.h"

namespace kerma {

class LinkKermaRTDevicePass : public llvm::ModulePass {
public:
  static char ID;
  LinkKermaRTDevicePass();
  bool runOnModule(llvm::Module &M) override;
};

};

#endif // KERMA_TRANSFORMS_LINK_KERMA_RT_DEVICE_PASS_H