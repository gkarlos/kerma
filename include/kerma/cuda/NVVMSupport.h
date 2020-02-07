//
// Created by gkarlos on 12-01-20.
//

#ifndef KERMA_STATIC_ANALYSIS_NVVMSUPPORT_H
#define KERMA_STATIC_ANALYSIS_NVVMSUPPORT_H

#include <kerma/cuda/CudaSupport.h>

#include <set>

namespace kerma {
namespace nvvm {

namespace instr {
/**
 * Supported LLVM instructions by NVVM
 * https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#instructions
 */

enum class Terminators {
  RET,
  BR,
  SWITCH,
  UNREACHABLE
};

enum class BinOps {
  ADD, FADD,
  SUB, FSUB,
  MUL, FMUL,
  UDIV,
  SDIV,
  FDIV,
  UREM,
  SREM,
  FREM
};

enum class BitwiseBinOps {
  SHL,
  LSHR,
  ASHR,
  AND,
  OR,
  XOR
};

enum class VectorOps {
  EXTRACTELEMENT,
  INSERTELEMENT,
  SHUFFLEVECTOR
};

enum class AggregateOps {
  EXTRACTVALUE,
  INSERTVALUE
};

enum class MemOps {
  /* *
   * 1. Always local address space
   * 2. Number of elements must be compile-time constant
   */
  ALLOCA,

  /* *
   * 1. Pointer must be <global> or <shared>, or <generic> that points to
   *    either global or shared address space
   */
  CMPXCHG,

  /* *
   * 1. nand not supported
   * 2. pointer bust me <global> or <shared>, or <generic> that points to
   *    either global or shared address space
   */
  ATOMICRMW,

  GETELEMENTPTR
};

enum class ConversionOps {
  TRUNC,
  ZEXT,
  SEXT,
  FPTRUNC,
  FPEXT,
  FPTOUI,
  FPTOSI,
  UITOFP,
  SITOFP,
  PTRTOINT,
  INTTOPTR,
  ADDRSPACECAST,
  BITCAST
};

enum class OtherOps {
  ICMP,
  FCMP,
  PHI,
  SELECT,
  CALL
};

} /// NAMESPACE instr

enum class intrinsics {
/**
 * Supported LLVM intrinsics by NVVM
 * https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#intrinsic-functions
 */

  // Standard C intrinsics
  LLVM_MEMCPY,
  LLVM_MEMMOVE,
  LLVM_MEMSET,
  LLVM_SQRT, /// float,double and vector float,double. sqrt.rn.f32 / sqrt.rn.f64
  LLVM_FMA,  /// float,double and vector float,double. fma.rn.f32 / fma.rn.f64

  // Bit Manipulation
  LLVM_BSWAP, /// i16, i32, i64
  LLVM_CTPOP, /// i8, i16, i32, i64 and vectors
  LLVM_CTLZ,  /// i8, i16, i32, i64 and vectors
  LLVM_CTTZ,  /// i8, i16, i32, i64 and vectors

  // Specialized Arithmetic
  LLVM_FMULADD,

  // Half Precision FP
  LLVM_CONVERT_TO_FP16_F32,
  LLVM_CONVERT_FROM_FP16_F32,
  LLVM_CONVERT_TO_FP16,
  LLVM_CONVERT_FROM_FP16,

  // Dbg
  LLVM_DBG_DECLARE,
  LLVM_DBG_VALUE,

  // Mem use markers
  LLVM_LIFETIME_START,
  LLVM_LIFETIME_END,
  LLVM_INVARIANT_START,
  LLVM_INVARIANT_END,

  // General
  LLVM_EXPECT,
  LLVM_DONOTHING,

  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-barrier
  LLVM_NVVM_BARRIER0,
  LLVM_NVVM_BARRIER0_POPC,
  LLVM_NVVM_BARRIER0_AND,
  LLVM_NVVM_BARRIER0_OR,
  LLVM_NVVM_MEMBAR_CTA,
  LLVM_NVVM_MEMBAR_GL,
  LLVM_NVVM_MEMBAR_SYS,

  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level
  LLVM_NVVM_BAR_WARP_SYNC,
  LLVM_NVVM_SHFL_SYNC_I32,
  LLVM_NVVM_VOTE_SYNC,
  LLVM_NVVM_MATCH_ANY_SYNC_I32,
  LLVM_NVVM_MATCH_ANY_SYNC_I64,
  LLVM_NVVM_MATCH_ALL_SYNC_I32,
  LLVM_NVVM_MATCH_ALL_SYNC_I64

  // TODO https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-matrix

} /// enum class intrinsics

static const std::set<kerma::cuda::Compute> SupportedComputes {
  cuda::Compute::CC30, cuda::Compute::CC32,
  cuda::Compute::CC35,
  cuda::Compute::CC37,
  cuda::Compute::CC50, cuda::Compute::CC52, cuda::Compute::CC53,
  cuda::Compute::CC60, cuda::Compute::CC61,
  cuda::Compute::CC70
};

static const std::set<kerma::cuda::AddressSpace> SupportedAddressSpaces {
  cuda::AddressSpace::GLOBAL,
  cuda::AddressSpace::SHARED,
  cuda::AddressSpace::CONSTANT
};

static const std::set<std::string> SupportedLLVMNamedMetadata {
  "!nvvm.annotations",
  "!nvvmir.version",
  "!llvm.dbg.cu",
  "!llvm.module.flags"
};

enum class SupportedLLVMLinkage {
  PRIVATE,
  INTERNAL

  EXTERNAL,

  COMMON,

  AVAILABLE_EXTERNALLY,
  LINKONCE,
  LINKONCE_ODR,
  WEAK,
  WEAK_ODR
};

struct AddressSpace {
  std::string name;
  int number;
};


namespace AddrSpace {
/// https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#address-space

/** CUDA C/C++ function     */
static const AddressSpace CODE     { "code",     0 };
/** Pointers in CUDA C/C++  */
static const AddressSpace GENERIC  { "generic",  0 };
/** CUDA C/C++ __device__   */
static const AddressSpace GLOBAL   { "global",   1 };
/** CUDA C/C++ __shared__   */
static const AddressSpace SHARED   { "shared",   3 };
/** CUDA C/C++ __constant__ */
static const AddressSpace CONSTANT { "constant", 4 };
/** CUDA C/C++ local        */
static const AddressSpace LOCAL    { "local",    5 };

} /// NAMESPACE AddrSpace

} /// NAMESPACE nvvm
} /// NAMESPACE kerma

#endif // KERMA_STATIC_ANALYSIS_NVVMSUPPORT_H
