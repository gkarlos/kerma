#include "kerma/NVVM/NVVMUtilities.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"

#include <mutex>

using namespace llvm;

namespace kerma {
namespace nvvm {

namespace {
  typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
  typedef std::map<const llvm::GlobalValue *, key_val_pair_t> global_val_annot_t;
  typedef std::map<const llvm::Module *, global_val_annot_t> per_module_annot_t;

  static llvm::ManagedStatic<per_module_annot_t> annotationCache;
  static llvm::sys::Mutex Lock;

  void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval) {
    std::lock_guard<sys::Mutex> Guard(Lock);
    assert(md && "Invalid mdnode for annotation");
    assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
    // start index = 1, to skip the global variable key
    // increment = 2, to skip the value for each property-value pairs
    for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
      // property
      const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
      assert(prop && "Annotation property not a string");

      // value
      ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(md->getOperand(i + 1));
      assert(Val && "Value operand not a constant int");

      std::string keyname = prop->getString().str();
      if (retval.find(keyname) != retval.end())
        retval[keyname].push_back(Val->getZExtValue());
      else {
        std::vector<unsigned> tmp;
        tmp.push_back(Val->getZExtValue());
        retval[keyname] = tmp;
      }
    }
  }

  void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) {
    std::lock_guard<sys::Mutex> Guard(Lock);
    NamedMDNode *NMD = m->getNamedMetadata("nvvm.annotations");
    if (!NMD)
      return;
    key_val_pair_t tmp;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      const MDNode *elem = NMD->getOperand(i);

      GlobalValue *entity =
          mdconst::dyn_extract_or_null<GlobalValue>(elem->getOperand(0));
      // entity may be null due to DCE
      if (!entity)
        continue;
      if (entity != gv)
        continue;

      // accumulate annotations for entity in tmp
      cacheAnnotationFromMD(elem, tmp);
    }

    if (tmp.empty()) // no annotations for this gv
      return;

    if ((*annotationCache).find(m) != (*annotationCache).end())
      (*annotationCache)[m][gv] = std::move(tmp);
    else {
      global_val_annot_t tmp1;
      tmp1[gv] = std::move(tmp);
      (*annotationCache)[m] = std::move(tmp1);
    }
  }

  bool findOneNVVMAnnotation(const llvm::GlobalValue *gv, const std::string &prop, unsigned &retval) {
    std::lock_guard<llvm::sys::Mutex> Guard(Lock);
    const Module *m = gv->getParent();
    if ((*annotationCache).find(m) == (*annotationCache).end())
      cacheAnnotationFromMD(m, gv);
    else if ((*annotationCache)[m].find(gv) == (*annotationCache)[m].end())
      cacheAnnotationFromMD(m, gv);
    if ((*annotationCache)[m][gv].find(prop) == (*annotationCache)[m][gv].end())
      return false;
    retval = (*annotationCache)[m][gv][prop][0];
    return true;
  }
} // anonymous namespace

// https://github.com/llvm/llvm-project/blob/master/llvm/lib/Target/NVPTX/NVPTXUtilities.cpp

bool isKernelFunction(const llvm::Function &F) {
  unsigned x = 0;
  bool retval = findOneNVVMAnnotation(&F, "kernel", x);
  if (!retval) {
    // There is no NVVM metadata, check the calling convention
    return F.getCallingConv() == llvm::CallingConv::PTX_Kernel;
  }
  return (x == 1);
}

bool isNVVMIntrinsic(const llvm::Function &F) {
  // We check if the intrinsic ID falls if the range of the NVVMIntrinsics enum
  // This means we may have to update the range if things change in that file
  return F.isIntrinsic() &&
         F.getIntrinsicID() >= Intrinsic::nvvm_add_rm_d && F.getIntrinsicID() <= Intrinsic::nvvm_wmma_m8n8k32_store_d_s32_row_stride;
}

bool isNVVMAtomic(const llvm::Function &F) {
  if ( !isNVVMIntrinsic(F))
    return false;

  switch( F.getIntrinsicID()) {
    case Intrinsic::nvvm_atomic_add_gen_f_cta:
    case Intrinsic::nvvm_atomic_add_gen_f_sys:
    case Intrinsic::nvvm_atomic_add_gen_i_cta:
    case Intrinsic::nvvm_atomic_add_gen_i_sys:
    case Intrinsic::nvvm_atomic_and_gen_i_cta:
    case Intrinsic::nvvm_atomic_and_gen_i_sys:
    case Intrinsic::nvvm_atomic_cas_gen_i_cta:
    case Intrinsic::nvvm_atomic_cas_gen_i_sys:
    case Intrinsic::nvvm_atomic_dec_gen_i_cta:
    case Intrinsic::nvvm_atomic_dec_gen_i_sys:
    case Intrinsic::nvvm_atomic_exch_gen_i_cta:
    case Intrinsic::nvvm_atomic_exch_gen_i_sys:
    case Intrinsic::nvvm_atomic_inc_gen_i_cta:
    case Intrinsic::nvvm_atomic_inc_gen_i_sys:
    case Intrinsic::nvvm_atomic_load_dec_32:
    case Intrinsic::nvvm_atomic_load_inc_32:
    case Intrinsic::nvvm_atomic_max_gen_i_cta:
    case Intrinsic::nvvm_atomic_max_gen_i_sys:
    case Intrinsic::nvvm_atomic_min_gen_i_cta:
    case Intrinsic::nvvm_atomic_min_gen_i_sys:
    case Intrinsic::nvvm_atomic_or_gen_i_cta:
    case Intrinsic::nvvm_atomic_or_gen_i_sys:
    case Intrinsic::nvvm_atomic_xor_gen_i_cta:
    case Intrinsic::nvvm_atomic_xor_gen_i_sys:
      return true;
    default:
      return false;
  }
}

bool isAtomic(const std::string& F) {
  return std::find(Atomics.begin(), Atomics.end(), F) != Atomics.end()
      || std::find(cc35::Atomics.begin(), cc35::Atomics.end(), F) != cc35::Atomics.end()
      || std::find(cc60::Atomics.begin(), cc60::Atomics.end(), F) != cc60::Atomics.end()
      || std::find(cc70::Atomics.begin(), cc70::Atomics.end(), F) != cc70::Atomics.end()
      || std::find(cc80::Atomics.begin(), cc80::Atomics.end(), F) != cc80::Atomics.end();
}

bool isIntrinsic(const std::string& F) {
  return std::find(Atomics.begin(), Intrinsics.end(), F) != Intrinsics.end()
      || std::find(cc35::Intrinsics.begin(), cc35::Intrinsics.end(), F) != cc35::Intrinsics.end()
      || std::find(cc60::Intrinsics.begin(), cc60::Intrinsics.end(), F) != cc60::Intrinsics.end()
      || std::find(cc70::Intrinsics.begin(), cc70::Intrinsics.end(), F) != cc70::Intrinsics.end()
      || std::find(cc80::Intrinsics.begin(), cc80::Intrinsics.end(), F) != cc80::Intrinsics.end();
}

const AddressSpace::Ty& getAddressSpaceWithId(int id) {
  switch(id) {
    case 1:
      return AddressSpace::Global;
    case 3:
      return AddressSpace::Shared;
    case 4:
      return AddressSpace::Constant;
    case 5:
      return AddressSpace::Local;
    case 0:
      return AddressSpace::Generic;
    case 2:
      return AddressSpace::Internal;
    case 7:
      return AddressSpace::LocalOrGlobal;
    case 8:
      return AddressSpace::LocalOrShared;
    default:
      return AddressSpace::Unknown;
  }
}

} // namespace nvvm
} // namespace kerma