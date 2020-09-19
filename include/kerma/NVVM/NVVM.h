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


namespace AddressSpace {

  namespace detail {
    struct AddressSpaceImpl {
    private:
      const char *name_;
    public:
      constexpr AddressSpaceImpl(const char *name, int id) : name_(name), ID(id) {}
      const int ID;
      operator std::string() const { return name_;}
      constexpr operator int() const { return ID; }
      constexpr bool operator==(const AddressSpaceImpl& other) { return ID == other.ID; }
      const std::string name() const { return name_; }
    };
  } // namespace detail

  constexpr detail::AddressSpaceImpl Global("global", 1);
  constexpr detail::AddressSpaceImpl Shared("shared", 3);
  constexpr detail::AddressSpaceImpl Constant("constant", 4);
  constexpr detail::AddressSpaceImpl Local("local", 5);
  constexpr detail::AddressSpaceImpl Generic("generic", 0);
  constexpr detail::AddressSpaceImpl Internal("internal",2);
  constexpr detail::AddressSpaceImpl Unknown("unknown", -1);
} // namespace AddressSpace

inline const AddressSpace::detail::AddressSpaceImpl& getAddressSpaceWithId(int id) {
  switch(id) {
    case AddressSpace::Global:
      return AddressSpace::Global;
    case AddressSpace::Shared:
      return AddressSpace::Shared;
    case AddressSpace::Constant:
      return AddressSpace::Constant;
    case AddressSpace::Local:
      return AddressSpace::Local;
    case AddressSpace::Generic:
      return AddressSpace::Generic;
    case AddressSpace::Internal:
      return AddressSpace::Internal;
    default:
      return AddressSpace::Unknown;
  }
}

} // namespace nvvm
} // namespace kerma

#endif // KERMA_NVVM_H