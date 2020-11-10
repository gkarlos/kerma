#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/Names.h"
#include "kerma/Base/Memory.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"

#include <llvm/ADT/SmallSet.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <string>

using namespace llvm;

namespace kerma {

const std::vector<Memory> &MemoryInfo::getForKernel(unsigned int KernelID) {
  auto it = M.find(KernelID);
  if ( it == M.end())
    throw std::runtime_error("MemoryInfo: Unknown kernel id " + std::to_string(KernelID));
  return it->second;
}

std::vector<Memory> MemoryInfo::getArgMemoriesForKernel(unsigned KernelID) {
  std::vector<Memory> Res;
  auto it = M.find(KernelID);
  for ( auto &Mem : it->second) {
    if ( Mem.isArgument())
      Res.push_back(Mem);
  }
  return Res;
}

Memory *MemoryInfo::getMemoryForArg(llvm::Argument *Arg) {
  auto *F = Arg->getParent();
  for ( auto &E : M) {
    for ( auto &Mem : E.second) {
      if ( Mem.isArgument() && Mem.getValue() == Arg)
        return &Mem;
    }
  }
  return nullptr;
}

unsigned MemoryInfo::getArgMemCount() {
  unsigned Res = 0;
  for ( auto &E : M) {
    for ( auto &Mem : E.second) {
      if ( isa<Argument>(Mem.getValue()))
        Res++;
    }
  }
  return Res;
}

unsigned MemoryInfo::getGVMemCount() {
  unsigned Res = 0;
  for ( auto &E : M) {
    for ( auto &Mem : E.second) {
      if ( isa<GlobalVariable>(Mem.getValue()))
        Res++;
    }
  }
  return Res;
}


// Pass

char DetectMemoriesPass::ID = 2;

DetectMemoriesPass::DetectMemoriesPass(std::vector<Kernel> &Kernels,
                                       bool SkipLocal)
    : ModulePass(ID), Kernels(Kernels), SkipLocal(SkipLocal) {}

static SmallSet<GlobalVariable *, 32> GetGlobalsUsedInKernel(Kernel &Kernel) {
  SmallSet<GlobalVariable *, 32> Globals;
  for (auto &BB : *Kernel.getFunction()) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction()->getName().startswith("llvm.dbg"))
          continue;
      for (Use &U : I.operands())
        for (auto &GV : Kernel.getFunction()->getParent()->globals())
          if (&GV == U->stripPointerCasts() &&
              !GV.getSection().startswith("llvm.metadata")) {
            Globals.insert(&GV);
            break;
          }
    }
  }
  return Globals;
}

static unsigned countArrayDims(ArrayType *Ty) {
  unsigned nest = 0;
  Type *tmp = Ty;
  while ( tmp && tmp->isArrayTy()) {
    ++nest;
    tmp = tmp->getArrayElementType();
  }
  return nest;
}

static Memory GetMemoryFromGlobal(const Kernel &Kernel, GlobalVariable &GV) {
  Memory Mem(Namer::GetNameForGlobal(&GV),
             nvvm::getAddressSpaceWithId(GV.getAddressSpace()));
  Mem.setKind(Memory::Kind::Global);
  Mem.setValue(&GV);
  Mem.addKernelUser(Kernel);

  auto *Ty = GV.getValueType();

  if ( Ty->isPointerTy()) {
    Mem.setKnownDim(Dim::None);
    Mem.setType(Ty->getPointerElementType());
  } else if ( Ty->isArrayTy()) {
    Dim dim;
    unsigned ndims = countArrayDims(cast<ArrayType>(Ty));
    if ( ndims == 1) {
      dim.x = Ty->getArrayNumElements();
      Mem.setType(Ty->getArrayElementType());
    } else if ( ndims == 2) {
      dim.y = Ty->getArrayNumElements();
      dim.x = Ty->getArrayElementType()->getArrayNumElements();
      Mem.setType(Ty->getArrayElementType()->getArrayElementType());
    } else {
      dim.z = Ty->getArrayNumElements();
      dim.y = Ty->getArrayElementType()->getArrayNumElements();
      dim.x = Ty->getArrayElementType()->getArrayElementType()->getArrayNumElements();
      Mem.setType(Ty->getArrayElementType()->getArrayElementType()->getArrayElementType());
    }
    Mem.setKnownDim(dim);
  } else {
    Mem.setType(Ty);
    Mem.setKnownDim(Dim::Unit);
  }

  return Mem;
}


static Memory GetMemoryFromArg(const Kernel &Kernel, Argument &Arg) {
  // At this point we know that Arg is a pointer
  // and is not ByVal
  Memory Mem(Arg.getName(), nvvm::AddressSpace::Global);
  Mem.setKind(Memory::Kind::Arg);
  Mem.setValue(&Arg);
  Mem.addKernelUser(Kernel);
  Mem.setKnownDim(Dim::None); // we cant know dim for ptrs
  Mem.setType(Arg.getType()->getPointerElementType());
  return Mem;
}

bool DetectMemoriesPass::runOnModule(llvm::Module &M) {
  for (auto &Kernel : Kernels) {
    MI.M[Kernel.getID()]; // make sure every kernel has an entry

    auto GlobalsUsed = GetGlobalsUsedInKernel(Kernel);
    for (auto *G : GlobalsUsed) {
      auto Mem = GetMemoryFromGlobal(Kernel, *G);
      // MI.Memories[Kernel.getID()][Mem.getValue()] = Mem;
      MI.M[Kernel.getID()].push_back(Mem);
    }
    for (auto &A : Kernel.getFunction()->args()) {
      if (A.hasAttribute(Attribute::ByVal) || !A.getType()->isPointerTy())
        continue;
      auto Mem = GetMemoryFromArg(Kernel, A);
      // MI.Memories[Kernel.getID()][Mem.getValue()] = Mem;
      MI.M[Kernel.getID()].push_back(Mem);
    }
  }
  return false;
}

} // namespace kerma