#include "kerma/Transforms/Instrument/MatInstrumenter.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/InferAddressSpaces.h"
#include "kerma/Analysis/MemoryAccessTree.h"
#include "kerma/Analysis/Names.h"
#include "kerma/Analysis/Typecheck.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/RT/Util.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Support/Parse.h"
#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"
#include "kerma/Transforms/Canonicalize/Canonicalizer.h"
#include "kerma/Transforms/Canonicalize/GepifyMem.h"
#include "kerma/Transforms/Canonicalize/SimplifyGEP.h"
#include "kerma/Transforms/Instrument/LinkDeviceRT.h"
#include "kerma/Transforms/StripAnnotations.h"
#include "kerma/Utils/LLVMMetadata.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>
#include <llvm/Pass.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

using namespace llvm;

namespace kerma {

static SmallSet<GlobalVariable *, 32> getGlobalsUsedInKernel(Function &Kernel) {
  SmallSet<GlobalVariable *, 32> Globals;

  for (auto &BB : Kernel) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction()->getName().startswith("llvm.dbg"))
          continue;
      for (Use &U : I.operands())
        for (auto &GV : Kernel.getParent()->globals())
          if (&GV == U->stripPointerCasts())
            if (!GV.getName().startswith("__kerma")) {
              Globals.insert(&GV);
              break;
            }
    }
  }
  return Globals;
}

static Type *getGlobalElementTy(Type *Ty) {
  if (!Ty->isAggregateType())
    return Ty;
  if (Ty->isStructTy())
    return Ty;

  if (auto *ty = dyn_cast<PointerType>(Ty))
    return ty->getElementType();

  auto *ElemTy = Ty;
  while (auto *elemty = dyn_cast<ArrayType>(ElemTy))
    ElemTy = elemty->getElementType();
  return ElemTy;
}

static unsigned int getSize(Module &M, Value *Ptr) {
  if (auto *PtrTy = dyn_cast<PointerType>(Ptr->getType()))
    return M.getDataLayout().getTypeStoreSize(PtrTy->getElementType());
  return 1;
}

static const std::string &getName(Value *V) {

  assert((isa<Argument>(V) || isa<GlobalVariable>(V) || isa<AllocaInst>(V)) &&
         "Object not an Arg/Loc/Glob");
  if (auto *Arg = dyn_cast<Argument>(V))
    return Namer::GetNameForArg(Arg, true);
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    return Namer::GetNameForGlobal(GV, true);
  if (auto *AI = dyn_cast<AllocaInst>(V))
    return Namer::GetNameForAlloca(AI, true);
  return Namer::None;
}

static bool isPtrByValArgument(Value *V) {
  if (!V)
    return false;
  if (auto *Arg = dyn_cast<Argument>(V))
    return Arg->getType()->isPointerTy() && Arg->hasAttribute(Attribute::ByVal);
  return false;
}

unsigned int MatInstrumenter::instrumentGlobalBaseAddresses(
    const Kernel &Kernel, Instruction *InsertAfter, Instruction *TraceStatus) {
  auto *Hook =
      Kernel.getFunction()->getParent()->getFunction("__kerma_rec_base");

  auto GVs = getGlobalsUsedInKernel(*Kernel.getFunction());

  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());

  unsigned int Count = 0;

  for (auto *GV : GVs) {
    auto name = getName(GV);
    if (GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] =
          IRB.CreateGlobalStringPtr(name, KermaGlobalSymbolPrefix);

    // IR globals are always pointers, get type of the pointee
    // If the type of the pointee is an array then get the type
    // of the elements of the array
    auto *GVPointeeTy = getGlobalElementTy(GV->getValueType());

    if (!GVPointeeTy) {
      WithColor::warning() << "Error recording base address for global: " << *GV
                           << '\n';
      continue;
    }

    // create a pointer to the the first element
    auto *GVPointeeTyPtr = PointerType::get(GVPointeeTy, 0);

    auto *Cast = AddrSpaceCastInst::CreatePointerCast(GV, GVPointeeTyPtr);
    Cast->insertAfter(InsertAfter);
    // auto *Cast = IRB.CreateAddrSpaceCast(GV, GVPointeeTyPtr);
    // auto PtrToStore = IRB.CreatePtrToInt(Cast,  IRB.getInt64Ty());
    auto *Base = PtrToIntInst::CreateBitOrPointerCast(Cast, IRB.getInt64Ty());
    Base->insertAfter(Cast);

    ArrayRef<Value *> Args(
        {/* status  */ TraceStatus,
         /* kernelid*/ ConstantInt::get(IRB.getInt8Ty(), Kernel.getID()),
         /* symbol  */ GlobalVariableForSymbol[name],
         /* addrspc */ ConstantInt::get(IRB.getInt8Ty(), GV->getAddressSpace()),
         /* base    */ Base});
    CallInst *CI = CallInst::Create(Hook, Args);
    CI->setCallingConv(CallingConv::PTX_Device);
    CI->insertAfter(Base);
    InsertAfter = CI;
    ++Count;
  }

  WithColor::note() << "Global base addresses recorded: " << Count << '/'
                    << GVs.size() << '\n';
  return Count;
}

unsigned int MatInstrumenter::instrumentArgBaseAddresses(
    const Kernel &Kernel, Instruction *InsertAfter, Instruction *TraceStatus) {
  auto *Hook =
      Kernel.getFunction()->getParent()->getFunction("__kerma_rec_base");

  if (!Hook)
    return false;

  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());
  IRB.SetInsertPoint(InsertAfter->getNextNode());

  unsigned int Count = 0, CountRec = 0;
  for (auto &Arg : Kernel.getFunction()->args()) {
    ++Count;

    if (Arg.hasAttribute(Attribute::ByVal))
      continue;

    auto ArgTy = Arg.getType();

    std::string name = getName(&Arg);

    if (GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] =
          IRB.CreateGlobalStringPtr(name, KermaGlobalSymbolPrefix);

    if (auto *ArgPtrTy = dyn_cast<PointerType>(ArgTy)) {
      // auto *Base = IRB.CreatePtrToInt(&Arg, IRB.getInt64Ty());

      auto *Base = PtrToIntInst::CreateBitOrPointerCast(&Arg, IRB.getInt64Ty());
      Base->insertAfter(InsertAfter);
      ArrayRef<Value *> Args(
          {/* status  */ TraceStatus,
           /* kernelid*/ ConstantInt::get(IRB.getInt8Ty(), Kernel.getID()),
           /* symbol  */ GlobalVariableForSymbol[name],
           /* addrspc */ ConstantInt::get(IRB.getInt8Ty(), 1),
           /* base    */ Base});
      // IRB.CreateCall(Hook, Args);
      CallInst *CI = CallInst::Create(Hook, Args);
      CI->insertAfter(Base);
      InsertAfter = CI;
      ++CountRec;
    }
  }

  WithColor::note() << "Arg base addresses recorded: " << CountRec << '/'
                    << Count << '\n';
  return Count;
}

bool MatInstrumenter::instrumentMeta(const Kernel &Kernel,
                                     Instruction *TraceStatus) {
  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());

  if (GlobalVariableForSymbol.find(Kernel.getDemangledName()) ==
      GlobalVariableForSymbol.end())
    GlobalVariableForSymbol[Kernel.getDemangledName()] =
        IRB.CreateGlobalStringPtr(Kernel.getDemangledName(),
                                  KermaGlobalSymbolPrefix);

  auto *Hook =
      Kernel.getFunction()->getParent()->getFunction("__kerma_rec_kernel");

  ArrayRef<Value *> Args(
      {/*status*/ TraceStatus,
       /* id   */ ConstantInt::get(IRB.getInt8Ty(), Kernel.getID()),
       /* name */ GlobalVariableForSymbol[Kernel.getDemangledName()]});

  auto *CI = CallInst::Create(Hook, Args);
  CI->setCallingConv(CallingConv::PTX_Device);
  CI->insertAfter(TraceStatus);

  // Record base addresses for arguments and globals
  // insert the calls after the __kerma_rec_kernel call
  if (!NullOffsets) {
    bool ArgChanges = instrumentArgBaseAddresses(Kernel, CI, TraceStatus);
    bool GlbChanges = instrumentGlobalBaseAddresses(Kernel, CI, TraceStatus);
  }

  return true;
}

/// Insert a trace status call at the beginning of the kernel
/// and return it.
static Instruction *instrumentTraceStatus(Kernel &Kernel) {
  auto *M = Kernel.getFunction()->getParent();
  auto *Hook = M->getFunction("__kerma_trace_status");
  auto *GV = M->getGlobalVariable(KermaTraceStatusSymbol);

  if (!Hook)
    return nullptr;

  auto *KernelID = ConstantInt::get(IntegerType::getInt32Ty(M->getContext()),
                                    Kernel.getID());

  IRBuilder<> IRB(Kernel.getFunction()->getEntryBlock().getFirstNonPHI());

  auto *Ptr = PointerType::get(GV->getValueType(), 0);
  auto *Cast = IRB.CreatePointerBitCastOrAddrSpaceCast(GV, Ptr);
  auto *GEP = IRB.CreateInBoundsGEP(
      GV->getValueType(), Cast,
      {ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0),
       ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0)});
  auto *TraceStatus = IRB.CreateCall(Hook, {GEP, KernelID});
  return TraceStatus;
}

/// Insert a call to __kerma_stop_tracing before
/// every return instruction encountered.
/// In the common case this should just insert one call
/// A better way is to probably run UnifyFunctionExitNodes first
/// (https://llvm.org/doxygen/UnifyFunctionExitNodes_8cpp_source.html
/// and then just get the (now unique) exit block and insert before
/// its terminator
static bool instrumentStopTracing(Kernel &Kernel) {
  auto *M = Kernel.getFunction()->getParent();
  auto *Hook = M->getFunction("__kerma_stop_tracing");
  auto *GV = M->getGlobalVariable(KermaTraceStatusSymbol);
  if (!Hook)
    return false;
  for (auto &BB : *Kernel.getFunction()) {
    for (auto &I : BB) {
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        auto *KernelID = ConstantInt::get(
            IntegerType::getInt32Ty(M->getContext()), Kernel.getID());

        IRBuilder<> IRB(Ret);
        auto *Ptr = PointerType::get(GV->getValueType(), 0);
        auto *Cast = IRB.CreatePointerBitCastOrAddrSpaceCast(GV, Ptr);
        auto *GEP = IRB.CreateInBoundsGEP(
            GV->getValueType(), Cast,
            {ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0),
             ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0)});
        if (!IRB.CreateCall(Hook, {GEP, KernelID}))
          return false;
      }
    }
  }
  return true;
}

static bool initInstrumentation(Module &M, const std::vector<Kernel> &Kernels) {

  /// This array keeps track of how many times a kernel function
  /// has been traced. Since (for now) we only want to trace a
  /// kernel once, a boolean array is enough.
  /// The size of the array is equal to the number of kernels we
  /// we detected.
  /// The first thing a thread does is bring the corresponding
  /// value in local memory and check it before at every call
  /// to a record function.
  /// At every exit point of a kernel, thread 0 will set the
  /// corresponding index to true.

  auto *Ty =
      ArrayType::get(IntegerType::getInt8Ty(M.getContext()), Kernels.size());

  auto *TraceStatusGV = new GlobalVariable(
      /* module    */ M,
      /* type      */ Ty,
      /* isConst   */ false,
      /* linkage   */ GlobalValue::ExternalLinkage,
      /* init      */ ConstantAggregateZero::get(Ty),
      /* name      */ KermaTraceStatusSymbol,
      /* insertpt  */ M.getGlobalVariable(KermaDeviceRTLinkedSymbol),
      /* threadloc */ GlobalValue::NotThreadLocal,
      /* addrspace */ 1,
      /* externinit*/ true);

  if (TraceStatusGV) {
    TraceStatusGV->setDSOLocal(true);
    TraceStatusGV->setAlignment(MaybeAlign(1));
  }

  return TraceStatusGV;
}

static Function *GetHook(Module &M, Mode Mode) {
  Function *Hook = nullptr;
  if (Mode == BLOCK) {
    Hook = M.getFunction("__rec_access_mat_b");
    if (!Hook)
      WithColor::warning() << "Failed to locate RT function __rec_access_mat_b";
  } else if (Mode == WARP) {
    Hook = M.getFunction("__rec_access_mat_w");
    if (!Hook)
      WithColor::warning() << "Failed to locate RT function __rec_access_mat_w";
  } else {
    Hook = M.getFunction("__rec_access_mat_t");
    if (!Hook)
      WithColor::warning() << "Failed to locate RT function __rec_access_mat_w";
  }
  return Hook;
}

static bool InstrumentAccesses(Kernel &K, Mode Mode, Value *TraceStatus,
                               bool NullOffsets) {
  auto *MAT = K.getMAT();
  auto *Hook = GetHook(*K.getFunction()->getParent(), Mode);
  auto Accesses = MAT->getAllAccesses();
  for (auto *A : Accesses) {
    errs() << "Instrumenting " << *A << '\n';
    IRBuilder<> IRB(A->getInst());
    auto *Ptr = A->getPtr();
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    assert(GEP);
    Value *Offset = IRB.CreatePtrToInt(Ptr, IRB.getInt64Ty());
    std::vector<Value *> Args{TraceStatus, IRB.getInt32(0) /* bid always 0 */};
    if (Mode == THREAD || Mode == WARP)
      Args.push_back(IRB.getInt32(0));        // wid or tid, again always 0
    Args.push_back(IRB.getInt32(A->getID())); // access_id

    if (NullOffsets) {
      std::vector<Value *> PtrIndices;
      for (auto it = GEP->idx_begin(); it != GEP->idx_end(); ++it)
        PtrIndices.push_back(*it);
      auto *NullOffset = GetElementPtrInst::Create(
          GEP->getPointerOperandType()->getPointerElementType(),
          Constant::getNullValue(GEP->getPointerOperandType()), PtrIndices);
      NullOffset->insertAfter(GEP);
      Args.push_back(IRB.CreatePtrToInt(NullOffset, IRB.getInt64Ty()));
    } else {
      Args.push_back(IRB.CreatePtrToInt(Offset, IRB.getInt64Ty()));
    }

    auto *Call = CallInst::Create(Hook, Args);
    Call->insertBefore(A->getInst());
  }
  return true;
}

bool MatInstrumenter::runOnModule(Module &M) {
  if (M.getTargetTriple().find("nvptx") == std::string::npos)
    return false;

  if (!getAnalysis<TypeCheckerPass>().moduleTypechecks())
    return false;

  InstrumentedFunctions.clear();
  GlobalVariableForSymbol.clear();

  if (!isDeviceRTLinked(M)) {
    WithColor::warning() << "KermaDeviceRT not found. Exiting..\n";
    return false;
  }
  if (!initInstrumentation(M, KI.getKernels())) {
    WithColor::warning() << " Failed to init instrumentation. Exiting..\n";
    return false;
  }

  bool Changed = false;

  for (auto &Kernel : KI.getKernels()) {
    bool KernelInstru = false;

    if (auto *TraceStatus = instrumentTraceStatus(Kernel)) {
      KernelInstru |= instrumentMeta(Kernel, TraceStatus);
      // KernelInstru &= instrumentAccesses(Kernel, TraceStatus);
      KernelInstru &=
          InstrumentAccesses(Kernel, Mode, TraceStatus, NullOffsets);
      KernelInstru &= instrumentStopTracing(Kernel);
    }

    if (!KernelInstru) {
      WithColor::warning() << " Failed to instrument kernel: "
                           << Kernel.getDemangledName() << '\n';
    }

    Changed |= KernelInstru;
  }

  return Changed;
}

void MatInstrumenter::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<TypeCheckerPass>();
  AU.setPreservesCFG();
}

Mode MatInstrumenter::getMode() { return Mode; }

char MatInstrumenter::ID = 4;

MatInstrumenter::MatInstrumenter(KernelInfo &KI, enum Mode Mode,
                                 bool NullOffsets)
    : ModulePass(ID), KI(KI), Mode(Mode), NullOffsets(NullOffsets) {}

bool MatInstrumenter::hasTargetFunction() { return !this->Targets.empty(); }

} // namespace kerma