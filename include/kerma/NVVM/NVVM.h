#ifndef KERMA_NVVM_H
#define KERMA_NVVM_H

#include <string>

namespace kerma {
namespace nvvm {

struct {
private:
  // const std::string v_ = "__cuda_builtin_gridDim_t";
  const std::string _ = "llvm.nvvm.read.ptx.sreg.nctaid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }

  // const std::string x = "__cuda_builtin_gridDim_t::__fetch_builtin_x()";
  // const std::string y = "__cuda_builtin_gridDim_t::__fetch_builtin_y()";
  // const std::string z = "__cuda_builtin_gridDim_t::__fetch_builtin_z()";
  const std::string x = "llvm.nvvm.read.ptx.sreg.nctaid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.nctaid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.nctaid.z";
} GridDim;

struct {
private:
  // const std::string v_ = "__cuda_builtin_blockDim_t";
  const std::string _ = "llvm.nvvm.read.ptx.sreg.ntid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }

  // const std::string x = "__cuda_builtin_blockDim_t::__fetch_builtin_x()";
  // const std::string y = "__cuda_builtin_blockDim_t::__fetch_builtin_y()";
  // const std::string z = "__cuda_builtin_blockDim_t::__fetch_builtin_z()";
  const std::string x = "llvm.nvvm.read.ptx.sreg.ntid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.ntid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.ntid.z";
} BlockDim;

struct {
private:
  // const std::string v_ = "__cuda_builtin_blockIdx_t";
  const std::string _ = "llvm.nvvm.read.ptx.sreg.ctaid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }

  // const std::string x = "__cuda_builtin_blockIdx_t::__fetch_builtin_x()";
  // const std::string y = "__cuda_builtin_blockIdx_t::__fetch_builtin_y()";
  // const std::string z = "__cuda_builtin_blockIdx_t::__fetch_builtin_z()";
  const std::string x = "llvm.nvvm.read.ptx.sreg.ctaid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.ctaid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.ctaid.z";
} BlockIdx;

struct {
private:
  // const std::string v_ = "__cuda_builtin_threadIdx_t";
  const std::string _ = "llvm.nvvm.read.ptx.sreg.tid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }

  // const std::string x = "__cuda_builtin_threadIdx_t::__fetch_builtin_x()";
  // const std::string y = "__cuda_builtin_threadIdx_t::__fetch_builtin_y()";
  // const std::string z = "__cuda_builtin_threadIdx_t::__fetch_builtin_z()";
  const std::string x = "llvm.nvvm.read.ptx.sreg.tid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.tid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.tid.z";
} ThreadIdx;

} // namespace nvvm
} // namespace kerma

#endif // KERMA_NVVM_H