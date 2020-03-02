#ifndef KERMA_SUPPORT_NVVM_H
#define KERMA_SUPPORT_NVVM_H

#include <llvm/IR/Module.h>
#include <kerma/Cuda/Cuda.h>

#include <string>

namespace kerma
{

class AddressSpace {
public:
  /** CUDA C/C++ function     */
  static AddressSpace CODE;
  /** Pointers in CUDA C/C++  */
  static AddressSpace GENERIC;
  /** CUDA C/C++ __device__   */
  static AddressSpace GLOBAL;
  /** CUDA C/C++ __shared__   */
  static AddressSpace SHARED;
  /** CUDA C/C++ __constant__ */
  static AddressSpace CONSTANT;
  /** CUDA C/C++ local        */
  static AddressSpace LOCAL;
  /** Unknown                 */
  static AddressSpace UNKNOWN;

  const std::string& getName();
  const int getCode();

  AddressSpace(const std::string& name, int code);

  bool operator==(AddressSpace& other);
  bool operator!=(AddressSpace& other);

private:
  const std::string name_;
  int code_;
};



/*
 * @brief Check if a Module is a Device Side LLVM IR Module
 * @param [in] module An LLVM IR Module
 */
bool isDeviceModule(llvm::Module& module);

/*
 * @brief Check if a Module is a Host Side LLVM IR Module
 * @param [in] module An LLVM IR Module
 */
bool isHostModule(llvm::Module& module);

/*
 * @brief Retrieve the side an LLVM IR is relevant for (host, device)
 * @param [in] module An LLVM IR Module
 */
CudaSide getIRModuleSide(llvm::Module &module);

}




#endif