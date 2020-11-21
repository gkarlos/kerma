#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/DetectMemories.h"
#include "kerma/Base/Assumption.h"
#include "kerma/Base/Kernel.h"
#include <algorithm>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <regex>
#include <spdlog/fmt/bundled/core.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kerma {

using namespace llvm;

AssumptionInfo &AssumptionInfo::add(llvm::Value *V, Assumption &A) {
  if (V) {
    if (auto *F = dyn_cast<Function>(V)) {
      if ( auto *LA = dyn_cast<LaunchAssumption>(&A))
        Launches[F] = *LA;
    } else if (auto *DA = dyn_cast<DimAssumption>(&A)) {
      Dims[V] = *DA;
    } else if (auto *VA = dyn_cast<ValAssumption>(&A)) {
      Vals[V] = *VA;
    }
  }
  return *this;
}

AssumptionInfo &AssumptionInfo::addLaunch(llvm::Function *F, LaunchAssumption &LA) {
  if (F)
    Launches[F] = LA;
  return *this;
}

std::vector<ValAssumption *> AssumptionInfo::getVals() {
  std::vector<ValAssumption *> Res;
  for (auto &E : Vals)
    Res.push_back(&E.second);
  return Res;
}

std::vector<DimAssumption *> AssumptionInfo::getDims() {
  std::vector<DimAssumption *> Res;
  for (auto &E : Dims)
    Res.push_back(&E.second);
  return Res;
}

std::vector<Assumption *> AssumptionInfo::getAll() {
  std::vector<Assumption *> Res;
  for (auto &E : Launches)
    Res.push_back(&E.second);
  for (auto &E : Vals)
    Res.push_back(&E.second);
  for (auto &E : Dims)
    Res.push_back(&E.second);
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

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// Parse a dim of the form z,y,x
static Dim parseDim(const std::string &S) {
  std::regex regex("\\,");
  std::vector<std::string> Components(
      std::sregex_token_iterator(S.begin(), S.end(), regex, -1),
      std::sregex_token_iterator());

  Dim D;
  if ( Components.size() == 1) {
    D = Dim(std::stol(Components[0]));
  } else if ( Components.size() == 2) {
    D = Dim(std::stol(Components[0]), std::stol(Components[1]));
  } else if ( Components.size() == 3) {
    D = Dim(std::stol(Components[0]), std::stol(Components[1]), std::stol(Components[2]));
  }

  return D;
}

static void getGlobalVarAssumptions(Module &M, KernelInfo &KI, MemoryInfo &MI,
                                    AssumptionInfo &AI) {
  // TODO: Implement me
  auto *GAs = M.getGlobalVariable("llvm.global.annotations");

  if (auto *GAs = M.getGlobalVariable("llvm.global.annotations")) {
    // Initializer is an array literal of struct literals (ConsantStruct)
    if (auto *GAInit = dyn_cast<ConstantArray>(GAs->getInitializer())) {
      for (auto &CAOp : GAInit->operands()) {
        if (auto *CS = dyn_cast<ConstantStruct>(&CAOp)) {
          // not sure if these structs can have less than 2 elements but lets
          // just be sure
          if (CS->getNumOperands() >= 2) {
            // The first operand is the global the annotation is about
            auto *G = CS->getOperand(0)->stripPointerCasts();
            // The second operand is the annotation value which is a GEP because
            // all globals are pointers, so get the GEP's pointer operand which
            // is the value itself
            if (GlobalVariable *A = dyn_cast<GlobalVariable>(
                    CS->getOperand(1)->getOperand(0))) {
              // get its initializer and extract the string
              if (ConstantDataArray *AInit =
                      dyn_cast<ConstantDataArray>(A->getInitializer())) {
                // Here we assume that all global annotations are kerma
                // annotations. For now this is ok but in general it is not. In
                // the future we should use a prefix for kerma annotations to
                // distringuish them from the others
                StringRef AS = AInit->getAsString();

                if (auto *F = dyn_cast<Function>(G)) {
                  if (auto *Kernel = KI.find(F)) {
                    auto Grid = parseDim(AS.substr(0, AS.find(':')));
                    auto Block = parseDim(AS.substr(AS.find(':') + 1, AS.size()));
                    LaunchAssumption LA(Grid, Block, Kernel->getFunction());
                    AI.addLaunch(Kernel->getFunction(), LA);
                  }
                } else if (auto *GV = dyn_cast<GlobalVariable>(G)) {
                  // GV is always a ptr so check element type
                  if( GV->getType()->getElementType()->isPointerTy()) {
                    if (auto *M = MI.getMemoryForVal(GV)) {
                      auto Dim = parseDim(AS);
                      M->setAssumedDim(Dim);
                      DimAssumption DA(Dim, *M);
                      AI.add(GV, DA);
                    } else {
                      llvm::errs() << " **warn** Ignored assumption for '" << GV->getName() << "'. Unused memory\n";
                    }
                  } else if (GV->getType()->getElementType()->isIntegerTy()) {
                    IAssumption IA(std::stoll(AS), GV);
                    AI.add(GV, IA);
                  } else if (GV->getType()->getElementType()->isFirstClassType()) {
                    FPAssumption FPA(std::stoll(AS), GV);
                    AI.add(GV, FPA);
                  } else {
                  }
                } else {
                  llvm::errs() << "**warning** Found annotation '" << AS
                               << "' for global that is neither a function nor "
                                  "a global variable\n";
                }
              }
            }
          }
        }
      }
    }
  }
}

static void getAssumptionForArg(ConstantDataArray *CDA, Argument *Arg,
                                KernelInfo &KI, MemoryInfo &MI,
                                AssumptionInfo &AI) {
  auto *Kernel = KI.find(Arg->getParent());
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

                auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer());
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
  // 1. Get assumptions for globals.
  getGlobalVarAssumptions(M, *KI, *MI, AI);
  // 2. Get assumptions for kernel Args
  getArgumentAssumptions(M, *KI, *MI, AI);
  return false;
}

} // namespace kerma