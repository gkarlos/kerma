#ifndef KERMA_SUPPORT_NVVM_H
#define KERMA_SUPPORT_NVVM_H

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

}




#endif