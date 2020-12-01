#include "kerma/Transforms/Materializer.h"
#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Base/Assumption.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Utils/LLVMShorthands.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/WithColor.h>

namespace kerma {

using namespace llvm;

static bool isBlockDimBuiltin(llvm::Function &F) {
  return demangle(F.getName()).find(nvvm::BlockDim) != std::string::npos;
}

static bool isGridDimBuiltin(llvm::Function &F) {
  return demangle(F.getName()).find(nvvm::GridDim) != std::string::npos;
}

static bool isThreadIdxBuiltin(llvm::Function &F) {
  return demangle(F.getName()).find(nvvm::ThreadIdx) != std::string::npos;
}

static bool isBlockIdxBuiltin(llvm::Function &F) {
  return demangle(F.getName()).find(nvvm::BlockIdx) != std::string::npos;
}

void MaterializeArguments(Kernel &K, AssumptionInfo &AI) {
  if (K.getFunction()) {
    for (auto &Arg : K.getFunction()->args()) {
      if (Arg.getType()->isPointerTy() || Arg.hasAttribute(Attribute::ByVal))
        continue;
      if (auto *A = AI.getForArg(&Arg)) {
        if (auto *FPA = dyn_cast<FPAssumption>(A)) {
          ConstantFP *Val = ConstantFP::get(K.getFunction()->getContext(),
                                            APFloat(FPA->getValue()));
          if (Arg.getType()->isFloatTy() || Arg.getType()->isDoubleTy())
            Arg.replaceNonMetadataUsesWith(Val);
          else
            WithColor::warning() << "MaterializeArguments: FP assumption "
                                 << FPA << " for non-fp arg " << Arg << '\n';
        } else if (auto *IA = dyn_cast<IAssumption>(A)) {
          if (auto *IntTy = dyn_cast<IntegerType>(Arg.getType())) {
            ConstantInt *Val = ConstantInt::get(IntTy, IA->getValue());
            Arg.replaceNonMetadataUsesWith(Val);
          } else
            WithColor::warning() << "MaterializeArguments: Int assumption "
                                 << IA << " for non-int arg " << Arg << '\n';
        } else {
          WithColor::warning() << "MaterializeArguments: Invalid assumption "
                               << *A << " for arg " << Arg << '\n';
        }
      } else {
        WithColor::warning()
            << "MaterializeArguments: No assumption found for arg " << Arg
            << '\n';
      }
    }
  }
}

void MaterializeDims(Kernel &K, AssumptionInfo &AI) {
  if (K.getFunction()) {
    auto *LA = K.getLaunchAssumption();
    if (!LA) {
      WithColor::warning()
          << "MaterializeDims: No launch assumptions for kernel '"
          << K.getName() << "'\n";
      return;
    }

    for (auto &BB : *K.getFunction()) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isBlockDimBuiltin(*Callee) || isGridDimBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::GridDim.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().x, 32));
            else if (DemangledCalleeName == nvvm::GridDim.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().y, 32));
            else if (DemangledCalleeName == nvvm::GridDim.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().x, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.x)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().x, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.y)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().y, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.z)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().z, 32));
            else
              assert(false && "MaterializeDims: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeIdx(Kernel &K, const Index &Bid, const Index &Tid) {
  if (K.getFunction()) {
    for (auto &BB : *K.getFunction()) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isBlockIdxBuiltin(*Callee) || isThreadIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::ThreadIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.x, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.y, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.z, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.x, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.y, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.z, 32));
            else
              assert(false && "MaterializeIdx: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeBody(Kernel &K, AssumptionInfo &AI, const Index &Bid, const Index &Tid) {
  if (K.getFunction()) {
    auto *LA = K.getLaunchAssumption();
    if (!LA) {
      WithColor::warning()
          << "MaterializeDims: No launch assumptions for kernel '"
          << K.getName() << "'\n";
      return;
    }

    for (auto &BB : *K.getFunction()) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isBlockDimBuiltin(*Callee) || isGridDimBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::GridDim.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().x, 32));
            else if (DemangledCalleeName == nvvm::GridDim.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().y, 32));
            else if (DemangledCalleeName == nvvm::GridDim.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), LA->getGrid().x, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.x)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().x, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.y)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().y, 32));
            else if (DemangledCalleeName == nvvm::BlockDim.z)
              I.replaceNonMetadataUsesWith(CreateUnsignedInt(
                  Callee->getContext(), LA->getBlock().z, 32));
            else
              assert(false && "MaterializeBody: Unknown dim call");
          } else if (isBlockIdxBuiltin(*Callee) ||
                     isThreadIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::ThreadIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.x, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.y, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Tid.z, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.x, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.y, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Bid.z, 32));
            else
              assert(false && "MaterializeBody: Unknown idx call");
          }
        }
      }
    }
  }
}

void MaterializeBlockIdx(Function *F, const Index &Idx) {
  if (F) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isBlockIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::BlockIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.x, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.y, 32));
            else if (DemangledCalleeName == nvvm::BlockIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.z, 32));
            else
              assert(false && "MaterializeBlockIdx: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeBlockIdx(llvm::Function *F, llvm::Value *Vz, llvm::Value *Vy,
                         llvm::Value *Vx) {
  if (F) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isBlockIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::BlockIdx.x)
              I.replaceNonMetadataUsesWith(Vx);
            else if (DemangledCalleeName == nvvm::BlockIdx.y)
              I.replaceNonMetadataUsesWith(Vy);
            else if (DemangledCalleeName == nvvm::BlockIdx.z)
              I.replaceNonMetadataUsesWith(Vz);
            else
              assert(false && "MaterializeBlockIdx: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeBlockIdx(Kernel &K, const Index &Idx) {
  MaterializeBlockIdx(K.getFunction(), Idx);
  // if (K.getFunction()) {
  //   for (auto &BB : *K.getFunction()) {
  //     for (auto &I : BB) {
  //       if (auto *CI = dyn_cast<CallInst>(&I)) {
  //         auto *Callee = CI->getCalledFunction();
  //         auto DemangledCalleeName = demangleFn(*Callee);
  //         if (isBlockIdxBuiltin(*Callee)) {
  //           if (DemangledCalleeName == nvvm::BlockIdx.x)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.x, 64));
  //           else if (DemangledCalleeName == nvvm::BlockIdx.y)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.y, 64));
  //           else if (DemangledCalleeName == nvvm::BlockIdx.z)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.z, 64));
  //           else
  //             assert(false && "MaterializeBlockIdx: this should never happen");
  //         }
  //       }
  //     }
  //   }
  // }
}

void MaterializeThreadIdx(Function *F, const Index &Idx) {
  if (F) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isThreadIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::ThreadIdx.x)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.x, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.y)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.y, 32));
            else if (DemangledCalleeName == nvvm::ThreadIdx.z)
              I.replaceNonMetadataUsesWith(
                  CreateUnsignedInt(Callee->getContext(), Idx.z, 32));
            else
              assert(false && "MaterializeIdx: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeThreadIdx(llvm::Function *F, llvm::Value *Vz, llvm::Value *Vy,
                          llvm::Value *Vx) {
  if (F) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          auto *Callee = CI->getCalledFunction();
          auto DemangledCalleeName = demangleFn(*Callee);
          if (isThreadIdxBuiltin(*Callee)) {
            if (DemangledCalleeName == nvvm::ThreadIdx.x)
              I.replaceNonMetadataUsesWith(Vx);
            else if (DemangledCalleeName == nvvm::ThreadIdx.y)
              I.replaceNonMetadataUsesWith(Vy);
            else if (DemangledCalleeName == nvvm::ThreadIdx.z)
              I.replaceNonMetadataUsesWith(Vz);
            else
              assert(false && "MaterializeBlockIdx: this should never happen");
          }
        }
      }
    }
  }
}

void MaterializeThreadIdx(Kernel &K, const Index &Idx) {
  MaterializeThreadIdx(K.getFunction(), Idx);
  // if (K.getFunction()) {
  //   for (auto &BB : *K.getFunction()) {
  //     for (auto &I : BB) {
  //       if (auto *CI = dyn_cast<CallInst>(&I)) {
  //         auto *Callee = CI->getCalledFunction();
  //         auto DemangledCalleeName = demangleFn(*Callee);
  //         if (isThreadIdxBuiltin(*Callee)) {
  //           if (DemangledCalleeName == nvvm::ThreadIdx.x)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.x, 64));
  //           else if (DemangledCalleeName == nvvm::ThreadIdx.y)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.y, 64));
  //           else if (DemangledCalleeName == nvvm::ThreadIdx.z)
  //             I.replaceNonMetadataUsesWith(
  //                 CreateUnsignedInt(Callee->getContext(), Idx.z, 64));
  //           else
  //             assert(false && "MaterializeIdx: this should never happen");
  //         }
  //       }
  //     }
  //   }
  // }
}

// Pass

char AssumptionMaterializerPass::ID = 45;

// void AssumptionMaterializerPass::getAnalysisUsage(
//     llvm::AnalysisUsage &AU) const {
//   AU.setPreservesAll();
// }

AssumptionMaterializerPass::AssumptionMaterializerPass(KernelInfo &KI,
                                                       AssumptionInfo &AI,
                                                       bool MaterializeDims,
                                                       bool MaterializeArgs)
    : ModulePass(ID), KI(KI), AI(AI), IgnoreDims(!MaterializeDims),
      IgnoreArgs(!MaterializeArgs) {
  if (IgnoreDims && IgnoreArgs && IgnoreIdx)
    WithColor::warning()
        << "Materializer created with F,F,F and will do no work!\n";
}

static void writeModuleToFile(Module &M, const std::string &Path) {
  std::error_code Err;
  raw_fd_ostream O(Path, Err);
  if (Err)
    throw std::runtime_error("writing module to file: " + Err.message());
  M.print(O, nullptr);
}

bool AssumptionMaterializerPass::runOnModule(llvm::Module &M) {
  for (auto &Kernel : KI.getKernels()) {
    if (!IgnoreArgs)
      MaterializeArguments(Kernel, AI);
    if (!IgnoreDims)
      MaterializeDims(Kernel, AI);
  }
  return true;
}

} // namespace kerma