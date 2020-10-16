#ifndef KERMA_TRANSFORMS_INSTRUMENT_LINK_DEVICE_RT_H
#define KERMA_TRANSFORMS_INSTRUMENT_LINK_DEVICE_RT_H

#include "llvm/Pass.h"

namespace kerma {

class LinkDeviceRTPass : public llvm::ModulePass {
private:
  std::string DeviceRTPath;
public:
  static char ID;
  LinkDeviceRTPass();
  LinkDeviceRTPass(std::string RTPath);
  void useDeviceRT(std::string RTPath);
  bool runOnModule(llvm::Module &M) override;
};

};

#endif // KERMA_TRANSFORMS_LINK_KERMA_DEVICE_RT_PASS_H