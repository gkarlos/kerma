#include "kerma/NVVM/NVVM.h"

namespace kerma {
namespace nvvm {

constexpr const AddressSpace::Ty AddressSpace::Unknown("unknown", -1);
constexpr const AddressSpace::Ty AddressSpace::Generic("generic", 0);
constexpr const AddressSpace::Ty AddressSpace::Global("global", 1);
constexpr const AddressSpace::Ty AddressSpace::Internal("internal",2);
constexpr const AddressSpace::Ty AddressSpace::Shared("shared", 3);
constexpr const AddressSpace::Ty AddressSpace::Constant("constant", 4);
constexpr const AddressSpace::Ty AddressSpace::Local("local", 5);
constexpr const AddressSpace::Ty AddressSpace::LocalOrGlobal("localOrGlobal", 7);
constexpr const AddressSpace::Ty AddressSpace::LocalOrShared("localOrShared", 8);


const std::vector<std::string> Symbols = {
  "threadIdx",
  "blockIdx",
  "blockDim",
  "gridDim"
};


namespace cc30 {
  const std::vector<std::string> Atomics{
    "atomicAdd(int*, int)",
    "atomicAdd(unsigned int*, unsigned int)",
    "atomicAdd(unsigned long long int*, unsigned long long int)"
    "atomicAdd(float*, float)",
    "atomicSub(int*, int)",
    "atomicSub(unsigned int*, unsigned int)",

    "atomicExch(int*, int)",
    "atomicExch(unsigned int*, unsigned int)",
    "atomicExch(unsigned long long int*, unsigned long long int)",
    "atomicExch(float*, float)",
    "atomicMin(int*, int)",
    "atomicMin(unsigned int*, unsigned int)",
    "atomicMax(int*, int)",
    "atomicMax(unsigned int*, unsigned int)",
    "atomicInc(unsigned int*, unsigned int)",
    "atomicDec(unsigned int*, unsigned int)",
    "atomicCAS(int*, int, int)",
    "atomicCAS(unsigned int*, unsigned int, unsigned int)",
    "atomicCAS(unsigned long long int*, unsigned long long int, unsigned long long int)",
    "atomicCas(unsigned short int*, unsigned short int, unsigned short int)"
    "atomicAnd(int*, int)",
    "atomicAnd(unsigned int*, unsigned int)",
    "atomicOr(int*, int)",
    "atomicOr(unsigned int*, unsigned int)",
    "atomicXor(int*, int)",
    "atomicXor(unsigned int*, unsigned int)",
  };

  const std::vector<std::string> Intrinsics{
    "__all_sync(unsigned int, int)",
    "__any_sync(unsigned int, int)",
    "__balloc_sync(unsigned int, int)",
    "__activemark()"
  };
}

namespace cc35 {
  const std::vector<std::string> Atomics = {
    "atomicMin(unsigned long long int*, unsigned long long int)",
    "atomicMax(unsigned long long int*, unsigned long long int)",
    "atomicAnd(unsigned long long int*, unsigned long long int)",
    "atomicOr(unsigned long long int*, unsigned long long int)",
    "atomicXor(unsigned long long int*, unsigned long long int)"
  };
  const std::vector<std::string> Intrinsics;
}

namespace cc60 {
  const std::vector<std::string> Atomics = {
    "atomicAdd(double*, double)",
    "atomicAdd(__half2*, __half2)",
    "atomicAdd(__nv_bfloat162*, __nv_bfloat162)"
  };
  const std::vector<std::string> Intrinsics;
}

namespace cc70 {
  const std::vector<std::string> Atomics = {
    "atomicAdd(__half*, __half)",
  };
  const std::vector<std::string> Intrinsics;
}
namespace cc80 {
  const std::vector<std::string> Atomics = {
    "atomicAdd(__nv_bfloat16*, __nv_bfloat16)"
  };
  const std::vector<std::string> Intrinsics;
}

const std::vector<std::string>& Atomics = cc30::Atomics;
const std::vector<std::string>& Intrinsics = cc30::Intrinsics;

} // namespace nvvm
} // namespace kerma

