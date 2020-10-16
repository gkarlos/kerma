#ifndef KERMA_NVVM_H
#define KERMA_NVVM_H

#include <string>
#include <vector>

namespace kerma {
namespace nvvm {

extern const std::vector<std::string> Symbols;

const struct {
private:
  const std::string _ = "llvm.nvvm.read.ptx.sreg.nctaid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }
  const std::string x = "llvm.nvvm.read.ptx.sreg.nctaid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.nctaid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.nctaid.z";
} GridDim;

const struct {
private:
  const std::string _ = "llvm.nvvm.read.ptx.sreg.ntid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }
  const std::string x = "llvm.nvvm.read.ptx.sreg.ntid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.ntid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.ntid.z";
} BlockDim;

const struct {
private:
  const std::string _ = "llvm.nvvm.read.ptx.sreg.ctaid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }
  const std::string x = "llvm.nvvm.read.ptx.sreg.ctaid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.ctaid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.ctaid.z";
} BlockIdx;

const struct {
private:
  const std::string _ = "llvm.nvvm.read.ptx.sreg.tid";
public:
  operator std::string() const { return _; }
  std::string operator+(std::string other) const { return *this + other; }
  const std::string x = "llvm.nvvm.read.ptx.sreg.tid.x";
  const std::string y = "llvm.nvvm.read.ptx.sreg.tid.y";
  const std::string z = "llvm.nvvm.read.ptx.sreg.tid.z";
} ThreadIdx;


namespace AddressSpace {
  namespace impl {

    struct AddressSpaceImpl {
    private:
      const char *name_;
    public:
      constexpr AddressSpaceImpl(const AddressSpaceImpl& Other) : AddressSpaceImpl(Other.name_, Other.ID) {}
      constexpr AddressSpaceImpl(const char *name, int id) : name_(name), ID(id) {}
      const int ID;
      operator std::string() const { return name_;}
      constexpr operator int() const { return ID; }
      constexpr bool operator==(const AddressSpaceImpl& other) { return ID == other.ID; }
      const std::string name() const { return name_; }
    }; // namespace impl
  } // namespace AddressSpace

  using Ty  = AddressSpace::impl::AddressSpaceImpl;

  extern const Ty Unknown;
  extern const Ty Generic;
  extern const Ty Global;
  extern const Ty Internal;
  extern const Ty Shared;
  extern const Ty Constant;
  extern const Ty Local;
  extern const Ty LocalOrGlobal;
  extern const Ty LocalOrShared;
}


namespace cc30 {
  extern const std::vector<std::string> Atomics;
  extern const std::vector<std::string> Intrinsics;
}

namespace cc35 {
  extern const std::vector<std::string> Atomics;
  extern const std::vector<std::string> Intrinsics;
}

namespace cc60 {
  extern const std::vector<std::string> Atomics;
  extern const std::vector<std::string> Intrinsics;
}

namespace cc70 {
  extern const std::vector<std::string> Atomics;
  extern const std::vector<std::string> Intrinsics;
}

namespace cc80 {
  extern const std::vector<std::string> Atomics;
  extern const std::vector<std::string> Intrinsics;
}

extern const std::vector<std::string>& Atomics;
extern const std::vector<std::string>& Intrinsics;


extern const std::vector<std::string> CudaAPI;


} // namespace nvvm
} // namespace kerma

#endif // KERMA_NVVM_H