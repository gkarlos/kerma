#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Assumption.h"
#include "kerma/Base/Kernel.h"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <spdlog/fmt/bundled/core.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kerma {

using namespace llvm;

AssumptionInfo &AssumptionInfo::add(llvm::Value *V, Assumption &A) {
  if (V) {
    if (auto *DA = dyn_cast<DimAssumption>(&A)) {
      Dims[V] = DA;
    } else if (auto *VA = dyn_cast<ValAssumption>(&A)) {
      Vals[V] = VA;
    }
  }
  return *this;
}

std::vector<ValAssumption *> AssumptionInfo::getVals() {
  std::vector<ValAssumption *> Res;
  for (auto &E : Vals)
    Res.push_back(E.second);
  return Res;
}

std::vector<DimAssumption *> AssumptionInfo::getDims() {
  std::vector<DimAssumption *> Res;
  for (auto &E : Dims)
    Res.push_back(E.second);
  return Res;
}

std::vector<Assumption *> AssumptionInfo::getAll() {
  std::vector<Assumption *> Res;
  for (auto &E : Vals)
    Res.push_back(E.second);
  for (auto &E : Dims)
    Res.push_back(E.second);
  return Res;
}

// Pass

char DetectAsumptionsPass::ID = 3;

#ifdef KERMA_OPT_PLUGIN
DetectAsumptionsPass::DetectAsumptionsPass()
    : DetectAsumptionsPass(nullptr, nullptr) {}
#endif

DetectAsumptionsPass::DetectAsumptionsPass(KernelInfo *KI, MemoryInfo *MI)
    : KI(KI), MI(MI), ModulePass(ID) {}

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

// Parse a dim of the form z,y,x
static Dim parseDim(const std::string &S) {
  std::regex regex("\\,");
  std::vector<std::string> Components(
      std::sregex_token_iterator(S.begin(), S.end(), regex, -1),
      std::sregex_token_iterator());
  Dim D(0);
  for (int i = Components.size() - 1; i >= 0; --i)
    D[i] = std::stol(Components[i]);
  return D;
}

static void getGlobalVarAssumptions(Module &M, KernelInfo &KI, MemoryInfo &MI,
                                    AssumptionInfo &AI) {
  // TODO: Implement me
}

static void getAssumptionForArg(ConstantDataArray *CDA, Argument *Arg,
                                KernelInfo &KI, MemoryInfo &MI,
                                AssumptionInfo &AI) {
  auto *Kernel = KI.getKernelByLLVMFn(Arg->getParent());
  assert(Kernel && "Assumption for Arg of non-kernel function!");

  try {
    if (Arg->getType()->isPointerTy() && !Arg->hasAttribute(Attribute::ByVal)) {
      if (auto *M = MI.getMemoryForArg(Arg)) {
        DimAssumption Ass(parseDim(CDA->getAsString()), *M);
        M->setAssumedDim(Ass.getDim());
        Ass.setMemory(*M);
        AI.add(Arg, Ass);
      }
    } else if (Arg->getType()->isIntegerTy()) {
      IAssumption Ass(std::stoll(CDA->getAsString()), Arg);
      AI.add(Arg, Ass);
    } else if (Arg->getType()->isFloatingPointTy()) {
      FPAssumption Ass(std::stod(CDA->getAsString()), Arg);
      AI.add(Arg, Ass);
    } else {
      throw;
    }
  } catch (...) {
    throw std::runtime_error("Failed to parse assumption " +
                             CDA->getAsString().str());
  }
}

static void getArgumentAssumptions(Module &M, KernelInfo &KI, MemoryInfo &MI,
                                   AssumptionInfo &AI) {
  for (auto &Kernel : KI.getKernels()) {
    if (!Kernel.getFunction())
      continue;

    auto Globals = GetGlobalsUsedInKernel(Kernel);

    // We assume that all nvvm.anotation are declared in the entry block
    for (auto &I : Kernel.getFunction()->getEntryBlock()) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction()->getName().startswith(
                "llvm.var.annotation")) {

          // Arg 0 is the address of the annotation string and it should be an
          // Alloca
          auto *Alloca =
              GetUnderlyingObject(CI->getArgOperand(0), M.getDataLayout());
          assert(isa<AllocaInst>(Alloca) &&
                 "Underlying oject of assumptions not an Alloca!");

          // Arg 1 is a pointer to the string value of the assumption
          auto *Assumption =
              GetUnderlyingObject(CI->getArgOperand(1), M.getDataLayout());
          assert(Assumption && "Could not find assumption value!");

          if (auto *A = dyn_cast<AllocaInst>(Alloca)) {
            for (auto *User : A->users()) {
              // We are looking for the first StoreInst globering this Alloca
              // It is important to run this pass before the Canonicalizer
              //
              // If the pass is run after the Canonicalizer we need to take
              // additional steps. In particular, Gepify turns the addr of
              // the StoreInst into a GEP. So search for that GEP first.
              // It should be the first one we find.
              //
              // For now assume the pass runs before the Canonicalizer
              // so just check for Arg or GV ptr at the StoreInst
              if (auto *SI = dyn_cast<StoreInst>(User)) {
                auto *Arg = dyn_cast<Argument>(SI->getOperand(0));
                assert(Arg && "Annotation is not for argument!");

                auto *GV = dyn_cast<GlobalVariable>(Assumption);
                assert(GV && "Assumption value not a global!");

                auto *CDA =
                    dyn_cast<ConstantDataArray>(GV->getInitializer());
                assert(CDA && "Failed to get Assumption Value initializer");

                getAssumptionForArg(CDA, Arg, KI, MI, AI);
                goto NEXT;
              }
              // }
            }
          }
        }
      }
    NEXT:
      continue;
    }
  }
}

bool DetectAsumptionsPass::runOnModule(llvm::Module &M) {
  // 1. Get assumptions for globals. This includes locally
  //    defined __shared__ variables
  getGlobalVarAssumptions(M, *KI, *MI, AI);
  // 2. Get assumptions for kernel Args
  getArgumentAssumptions(M, *KI, *MI, AI);
  return false;
}

} // namespace kerma