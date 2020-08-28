#ifndef KERMA_NVVM_H
#define KERMA_NVVM_H

#include <string>

namespace kerma {
namespace nvvm {

struct {
private:
  const std::string v_ = "__cuda_builtin_gridDim_t";
public:
  operator std::string() const { return v_; }
  std::string operator+(std::string other) const { return *this + other; }

  const std::string x = "__cuda_builtin_gridDim_t::__fetch_builtin_x()";
  const std::string y = "__cuda_builtin_gridDim_t::__fetch_builtin_y()";
  const std::string z = "__cuda_builtin_gridDim_t::__fetch_builtin_z()";
} GridDim;

struct {
private:
  const std::string v_ = "__cuda_builtin_blockDim_t";
public:
  operator std::string() const { return v_; }
  std::string operator+(std::string other) const { return *this + other; }

  const std::string x = "__cuda_builtin_blockDim_t::__fetch_builtin_x()";
  const std::string y = "__cuda_builtin_blockDim_t::__fetch_builtin_y()";
  const std::string z = "__cuda_builtin_blockDim_t::__fetch_builtin_z()";
} BlockDim;

struct {
private:
  const std::string v_ = "__cuda_builtin_blockIdx_t";
public:
  operator std::string() const { return v_; }
  std::string operator+(std::string other) const { return *this + other; }

  const std::string x = "__cuda_builtin_blockIdx_t::__fetch_builtin_x()";
  const std::string y = "__cuda_builtin_blockIdx_t::__fetch_builtin_y()";
  const std::string z = "__cuda_builtin_blockIdx_t::__fetch_builtin_z()";
} BlockIdx;

struct {
private:
  const std::string v_ = "__cuda_builtin_threadIdx_t";
public:
  operator std::string() const { return v_; }
  std::string operator+(std::string other) const { return *this + other; }

  const std::string x = "__cuda_builtin_threadIdx_t::__fetch_builtin_x()";
  const std::string y = "__cuda_builtin_threadIdx_t::__fetch_builtin_y()";
  const std::string z = "__cuda_builtin_threadIdx_t::__fetch_builtin_z()";
} ThreadIdx;

} // namespace nvvm
} // namespace kerma

#endif // KERMA_NVVM_H