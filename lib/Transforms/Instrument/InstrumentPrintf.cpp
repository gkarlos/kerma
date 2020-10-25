#include "kerma/Transforms/Instrument/InstrumentPrintf.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/InferDimensions.h"
#include "kerma/Analysis/InferAddressSpaces.h"
#include "kerma/Analysis/Names.h"
#include "kerma/Analysis/Typecheck.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/RT/Util.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Support/Parse.h"
#include "kerma/Utils/LLVMMetadata.h"
#include "kerma/Transforms/Instrument/LinkDeviceRT.h"
#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Pass.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/WithColor.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>


using namespace llvm;

#ifdef KERMA_OPT_PLUGIN
namespace {

// Set up some cl args for Opt
cl::OptionCategory InstruOptionCategory("Kerma Instrument Memory Operations pass Options (--kerma-instru)");

cl::opt<std::string> InstruTarget("kerma-instru-target", cl::Optional, cl::desc("Target specific kernel function"),
                                  cl::value_desc("kernel_name[,kernel_name]"), cl::cat(InstruOptionCategory), cl::init(""));
cl::opt<kerma::Mode> InstruMode("kerma-instru-mode", cl::Optional, cl::desc("Instrumentation mode"),
                                cl::values(
                                  clEnumValN(kerma::Mode::BLOCK_MODE, "block", "Instrument all threads in block 0"),
                                  clEnumValN(kerma::Mode::WARP_MODE, "warp", "Instrument all threads in warp 0 of block 0"),
                                  clEnumValN(kerma::Mode::THREAD_MODE, "thread", "Instrument thread 0 in block 0")
                                ), cl::init(kerma::Mode::BLOCK_MODE), cl::cat(InstruOptionCategory));

}
#endif

namespace kerma {

SmallSet<GlobalVariable *, 32> getGlobalsUsedInKernel(Function &Kernel) {
  SmallSet<GlobalVariable *, 32> Globals;

  for ( auto &BB : Kernel) {
    for ( auto &I : BB) {
      if ( auto *CI = dyn_cast<CallInst>(&I))
        if ( CI->getCalledFunction()->getName().startswith("llvm.dbg"))
          continue;
      for (Use &U : I.operands())
        for ( auto &GV : Kernel.getParent()->globals())
          if ( &GV == U->stripPointerCasts())
            if ( !GV.getName().startswith("__kerma")) {
              Globals.insert(&GV);
              break;
            }
    }
  }

  return Globals;
}

static Type* getGlobalElementTy(Type *Ty) {
  if ( !Ty->isAggregateType())
    return Ty;
  if ( Ty->isStructTy())
    return Ty;

  if ( auto *ty = dyn_cast<PointerType>(Ty))
    return ty->getElementType();

  auto *ElemTy = Ty;
  while ( auto *elemty = dyn_cast<ArrayType>(ElemTy))
    ElemTy = elemty->getElementType();
  return ElemTy;
}


static unsigned int getSize(Module &M, Value *Ptr) {
  if ( auto *PtrTy = dyn_cast<PointerType>(Ptr->getType()))
    return M.getDataLayout().getTypeStoreSize(PtrTy->getElementType());
  return 1;
}

static const std::string& getName(Value *V) {
  assert((isa<Argument>(V) || isa<GlobalVariable>(V) || isa<AllocaInst>(V)) && "Object not an Arg/Loc/Glob");
  if ( auto *Arg = dyn_cast<Argument>(V))
    return Namer::GetNameForArg(Arg, true);
  if ( auto *GV = dyn_cast<GlobalVariable>(V))
    return Namer::GetNameForGlobal(GV, true);
  if ( auto *AI = dyn_cast<AllocaInst>(V))
    return Namer::GetNameForAlloca(AI, true);
  return Namer::None;
}

static bool isPtrByValArgument(Value *V) {
  if (!V) return false;
  if ( auto *Arg = dyn_cast<Argument>(V))
    return Arg->getType()->isPointerTy() && Arg->hasAttribute(Attribute::ByVal);
  return false;
}

unsigned int
InstrumentPrintfPass::instrumentGlobalBaseAddresses(const Kernel& Kernel,  Instruction *InsertAfter, Instruction *TraceStatus) {
  auto *Hook = Kernel.getFunction()->getParent()->getFunction("__kerma_rec_base");

  auto GVs = getGlobalsUsedInKernel(*Kernel.getFunction());

  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());

  unsigned int Count = 0;

  for ( auto *GV : GVs) {
    auto name = getName(GV);
    if ( GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] = IRB.CreateGlobalStringPtr(name, KermaGlobalSymbolPrefix);

    // IR globals are always pointers, get type of the pointee
    // If the type of the pointee is an array then get the type
    // of the elements of the array
    auto *GVPointeeTy = getGlobalElementTy(GV->getValueType());

    if ( !GVPointeeTy) {
      WithColor::warning() << "Error recording base address for global: " << *GV << '\n';
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

    ArrayRef<Value*> Args({ /* status  */ TraceStatus,
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

  WithColor::note() << "Global base addresses recorded: " << Count << '/' << GVs.size() << '\n';
  return Count;
}

unsigned int
InstrumentPrintfPass::instrumentArgBaseAddresses(const Kernel& Kernel, Instruction *InsertAfter, Instruction *TraceStatus) {
  auto *Hook = Kernel.getFunction()->getParent()->getFunction("__kerma_rec_base");

  if ( !Hook) return false;

  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());
  IRB.SetInsertPoint(InsertAfter->getNextNode());

  unsigned int Count = 0, CountRec = 0;
  for ( auto& Arg : Kernel.getFunction()->args()) {
    ++Count;

    if ( Arg.hasAttribute(Attribute::ByVal))
      continue;

    auto ArgTy = Arg.getType();

    std::string name = getName(&Arg);

    if ( GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] = IRB.CreateGlobalStringPtr(name, KermaGlobalSymbolPrefix);

    if ( auto *ArgPtrTy = dyn_cast<PointerType>(ArgTy)) {
      // auto *Base = IRB.CreatePtrToInt(&Arg, IRB.getInt64Ty());

      auto *Base = PtrToIntInst::CreateBitOrPointerCast(&Arg, IRB.getInt64Ty());
      Base->insertAfter(InsertAfter);
      ArrayRef<Value*> Args({ /* status  */ TraceStatus,
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

  WithColor::note() << "Arg base addresses recorded: " << CountRec << '/' << Count << '\n';
  return Count;
}

bool InstrumentPrintfPass::instrumentMeta(const Kernel& Kernel, Instruction *TraceStatus) {
  IRBuilder<> IRB(Kernel.getFunction()->front().getFirstNonPHI());

  if ( GlobalVariableForSymbol.find(Kernel.getDemangledName()) == GlobalVariableForSymbol.end())
    GlobalVariableForSymbol[Kernel.getDemangledName()] = IRB.CreateGlobalStringPtr(Kernel.getDemangledName(), KermaGlobalSymbolPrefix);

  auto *Hook = Kernel.getFunction()->getParent()->getFunction("__kerma_rec_kernel");

  ArrayRef<Value*> Args({/*status*/ TraceStatus,
                         /* id   */ ConstantInt::get(IRB.getInt8Ty(), Kernel.getID()),
                         /* name */ GlobalVariableForSymbol[Kernel.getDemangledName()]});

  auto *CI = CallInst::Create(Hook, Args);
  CI->setCallingConv(CallingConv::PTX_Device);
  CI->insertAfter(TraceStatus);

  // Record base addresses for arguments and globals
  // insert the calls after the __kerma_rec_kernel call
  bool ArgChanges = instrumentArgBaseAddresses(Kernel, CI, TraceStatus);
  bool GlbChanges = instrumentGlobalBaseAddresses(Kernel, CI, TraceStatus);

  return GlbChanges | ArgChanges;
}

static bool isIntermediateObject(Value *Obj) {
  assert(Obj);
  return !isa<Argument>(Obj) && !isa<GlobalVariable>(Obj) && !isa<AllocaInst>(Obj);
}

bool isPointerToStruct(Type *Ty) {
  return Ty && isa<PointerType>(Ty) && dyn_cast<PointerType>(Ty)->getElementType()->isStructTy();
}

bool isArrayOfStructs(Type *Ty) {
  return Ty && isa<ArrayType>(Ty) && dyn_cast<ArrayType>(Ty)->getElementType()->isStructTy();
}

MemCpyInst *isMemCpy(Instruction *I) { return dyn_cast_or_null<MemCpyInst>(I); }


static Function * getHook(Module& M, AccessType AT, Mode Mode) {
  if ( Mode == BLOCK_MODE)
    return AT == AccessType::Memcpy? M.getFunction("__kerma_rec_copy_b")
                                   : M.getFunction("__kerma_rec_access_b");
  else if ( Mode == WARP_MODE)
    return AT == AccessType::Memcpy? M.getFunction("__kerma_rec_copy_w")
                                   : M.getFunction("__kerma_rec_access_w");
  else
    return AT == AccessType::Memcpy? M.getFunction("__kerma_rec_copy_t")
                                   : M.getFunction("__kerma_rec_access_t");
}

bool
InstrumentPrintfPass::instrumentCopy(const Kernel& K, MemCpyInst *I, Instruction *TraceStatus) {
  auto Err = [I]{ WithColor::warning() << " Failed to instrument copy: " << *I << '\n'; return false;};

  auto *M = K.getFunction()->getParent();

  auto *SourcePtr = I->getRawSource();
  auto *DestPtr = I->getRawDest();
  if ( !SourcePtr || !DestPtr) return Err();

  auto *SourceObj = GetUnderlyingObject(SourcePtr, M->getDataLayout());
  auto *DestObj = GetUnderlyingObject(DestPtr, M->getDataLayout());
  if ( !SourceObj || !DestObj) return Err();

  bool SourceLocal = (isa<AllocaInst>(SourceObj) || isPtrByValArgument(SourceObj));
  bool DestLocal = (isa<AllocaInst>(DestObj) || isPtrByValArgument(DestObj));

  if ( auto *Hook = getHook(*M, AccessType::Memcpy, Mode)) {
    IRBuilder<> IRB(I);
    ArrayRef<Value*> Args;
    std::string SourceName, DestName;

    if ( !SourceLocal) {
      SourceName = getName(SourceObj);
      if ( GlobalVariableForSymbol.find(SourceName) == GlobalVariableForSymbol.end())
        GlobalVariableForSymbol[SourceName] = IRB.CreateGlobalStringPtr(SourceName, "__kerma_sym_" + SourceName);
    }

    if ( !DestLocal) {
      DestName = getName(DestObj);
      if ( GlobalVariableForSymbol.find(DestName) == GlobalVariableForSymbol.end())
        GlobalVariableForSymbol[DestName] = IRB.CreateGlobalStringPtr(DestName, "__kerma_sym_" + DestName);
    }

    Value *SourceOffset = SourceLocal ? ConstantInt::get(IRB.getInt64Ty(), 0) // just 0 if local
                                      : IRB.CreatePtrToInt(SourcePtr,  IRB.getInt64Ty());

    Value *DestOffset = DestLocal ? ConstantInt::get(IRB.getInt64Ty(), 0) // just 0 if local
                                  : IRB.CreatePtrToInt(DestPtr,  IRB.getInt64Ty());

    SourceLoc Loc;
    if ( auto& DL = I->getDebugLoc()) {
      Loc.line = DL->getLine();
      Loc.col = DL.getCol();
    }

    // IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())

    if ( Mode == BLOCK_MODE)
      Args = {/*status*/ TraceStatus,
              /* bid  */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line */ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col  */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* sname*/ SourceLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                    : GlobalVariableForSymbol[SourceName],
              /* soff */ SourceOffset,
              /* dname*/ DestLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                  : GlobalVariableForSymbol[DestName],
              /* doff */ DestOffset,
              /* sz   */ IRB.CreateZExt(I->getLength(), IRB.getInt32Ty())};
    else if ( Mode == WARP_MODE)
      Args = {/*status*/ TraceStatus,
              /* bid  */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* wid  */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line */ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col  */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* sname*/ SourceLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                    : GlobalVariableForSymbol[SourceName],
              /* soff */ SourceOffset,
              /* dname*/ DestLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                  : GlobalVariableForSymbol[DestName],
              /* doff */ DestOffset,
              /* sz   */ IRB.CreateZExt(I->getLength(), IRB.getInt32Ty())};
    else // THREAD_MODE
      Args = {/*status*/ TraceStatus,
              /* bid  */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* tid  */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line */ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col  */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* sname*/ SourceLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                    : GlobalVariableForSymbol[SourceName],
              /* soff */ SourceOffset,
              /* dname*/ DestLocal? IRB.CreatePtrToInt(ConstantInt::get(IRB.getInt8Ty(), 0), IRB.getInt8PtrTy())
                                  : GlobalVariableForSymbol[DestName],
              /* doff */ DestOffset,
              /* sz   */ IRB.CreateZExt(I->getLength(), IRB.getInt32Ty())};

    if (auto *CI = CallInst::Create(Hook, Args, "", I)) {
      CI->setCallingConv(CallingConv::PTX_Device);
      return true;
    }
  }

  return Err();
}

bool
InstrumentPrintfPass::insertCallForAccess(AccessType AT,  const Kernel& Kernel,
                                          std::string Name, Value *Ptr, unsigned int Size,
                                          SourceLoc& Loc, Instruction *InsertBefore, Value *TraceStatus) {
  IRBuilder<> IRB(InsertBefore);

  // Check if a symbol for this name exists and create it if not
  if ( GlobalVariableForSymbol.find(Name) == GlobalVariableForSymbol.end())
    GlobalVariableForSymbol[Name] = IRB.CreateGlobalStringPtr(Name, "__kerma_sym#" + Name);

  if ( Function *Hook = getHook(*Kernel.getFunction()->getParent(), AT, Mode) ) {
    ArrayRef<Value*> Args;
    Value *Offset = IRB.CreatePtrToInt(Ptr,  IRB.getInt64Ty());

    if ( Mode == BLOCK_MODE)
      Args = {/* stat*/ TraceStatus,
              /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AT),
              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* name*/ GlobalVariableForSymbol[Name],
              /* off */ Offset,
              /* sz  */ ConstantInt::get(IRB.getInt32Ty(), Size)};
    else if ( Mode == WARP_MODE)
      Args = {/* stat*/ TraceStatus,
              /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AT),
              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* wid */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* name*/ GlobalVariableForSymbol[Name],
              /* off */ Offset,
              /* sz  */ ConstantInt::get(IRB.getInt32Ty(), Size)};
    else // THREAD_MODE
      Args = {/* stat*/ TraceStatus,
              /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AT),
              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* tid */ ConstantInt::get(IRB.getInt32Ty(), 0),
              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
              /* name*/ GlobalVariableForSymbol[Name],
              /* off */ Offset,
              /* sz  */ ConstantInt::get(IRB.getInt32Ty(), Size)};

    CallInst *CI = CallInst::Create(Hook, Args, "", InsertBefore);
    CI->setCallingConv(CallingConv::PTX_Device);
    return CI;
  }

  return false;
}

// Perform instrumenentation for a memory access
bool InstrumentPrintfPass::instrumentAccess(const Kernel& K, Instruction *I, Instruction *TraceStatus) {
  assert(!isa<MemCpyInst>(I) && "instrumentAccess cannot handle memcpy"); // sanity check

  auto *M = K.getFunction()->getParent();
  Value *Ptr = nullptr;
  Value *Obj = nullptr;
  AccessType Type;

  // Put this case here because a memcpy is actually
  // a call to llvm.memcpy... and it will match the
  // CallInst case below. So we want to catch it before that.
  if ( auto *MemCpy = dyn_cast<MemCpyInst>(I))
    return instrumentCopy(K, MemCpy, TraceStatus);

  if ( auto *LI = dyn_cast<LoadInst>(I)) {
    Ptr = LI->getPointerOperand(); //dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
    Type = AccessType::Load;
  }
  else if ( auto *SI = dyn_cast<StoreInst>(I)) {
    Ptr = SI->getPointerOperand(); //dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
    Type = AccessType::Store;
  }
  else if ( auto *CI = dyn_cast<CallInst>(I)) {
    if ( nvvm::isAtomicFunction(*CI->getCalledFunction())
      || nvvm::isReadOnlyCacheFunction(*CI->getCalledFunction()))
    {
      Ptr = dyn_cast<GetElementPtrInst>(CI->getArgOperand(0));
      Type = AccessType::Atomic;
    }
  }

  if ( !Ptr) {
    WithColor::warning() << "Could not locate pointer of: " << *I << '\n';
    return false;
  }

  Obj = GetUnderlyingObject(Ptr, K.getFunction()->getParent()->getDataLayout());

  if ( !Obj) {
    WithColor::warning() << "Could not locate underlying object of: " << *I << '\n';
    return false;
  }

  // At this point we have a Unary access to memory, and we
  // have an underlying object. For now, if this access is
  // performed to local memory we ignore it and log the event
  if ( isa<AllocaInst>(Obj) || isPtrByValArgument(Obj)) {
    WithColor::remark() << " ignoring local access: " << *I << '\n';
    return false;
  }

  // Try get some source info
  // Later one we will replace this with something like
  // a stmt ID, after we extract stmts with clang
  SourceLoc Loc;
  if ( auto& DL = I->getDebugLoc()) {
    Loc.line = DL->getLine();
    Loc.col = DL.getCol();
  }

  auto Success = insertCallForAccess(Type, K, getName(Obj), Ptr, getSize(*M, Ptr), Loc, I, TraceStatus);

  if ( !Success)
    WithColor::warning() << "Failed to instrument: " << *I << '\n';

  return Success;
}

bool InstrumentPrintfPass::instrumentAccesses(const Kernel& Kernel, Instruction *TraceStatus) {
  /// Just filter out the non-interesting instructions
  bool Changed = false;
  for ( auto &BB : *Kernel.getFunction()) {
    for ( auto &I : BB) {
      if ( isa<LoadInst>(&I)
        || isa<StoreInst>(&I)
        || isa<CallInst>(&I)
            && ( nvvm::isAtomicFunction( *dyn_cast<CallInst>(&I)->getCalledFunction())
              || nvvm::isReadOnlyCacheFunction( *dyn_cast<CallInst>(&I)->getCalledFunction())))
        Changed |= instrumentAccess(Kernel, &I, TraceStatus);
      else if (isa<MemCpyInst>(&I))
        Changed |= instrumentCopy(Kernel, dyn_cast<MemCpyInst>(&I), TraceStatus);
    }
  }
  return Changed;
}

/// Insert a trace status call at the beginning of the kernel
/// and return it.
static Instruction * instrumentTraceStatus(Kernel& Kernel) {
  auto *M = Kernel.getFunction()->getParent();
  auto *Hook = M->getFunction("__kerma_trace_status");
  auto *GV = M->getGlobalVariable(KermaTraceStatusSymbol);

  if ( !Hook) return nullptr;

  auto *KernelID = ConstantInt::get( IntegerType::getInt32Ty(M->getContext()),
                                     Kernel.getID());

  IRBuilder<> IRB(Kernel.getFunction()->getEntryBlock().getFirstNonPHI());

  auto *Ptr = PointerType::get(GV->getValueType(), 0);
  auto *Cast = IRB.CreatePointerBitCastOrAddrSpaceCast(GV, Ptr);
  auto *GEP = IRB.CreateInBoundsGEP( GV->getValueType(), Cast, {
                                     ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0),
                                     ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0) });
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
static bool instrumentStopTracing(Kernel& Kernel) {
  auto *M = Kernel.getFunction()->getParent();
  auto *Hook = M->getFunction("__kerma_stop_tracing");
  auto *GV = M->getGlobalVariable(KermaTraceStatusSymbol);
  if ( !Hook) return false;
  for ( auto &BB : *Kernel.getFunction()) {
    for ( auto &I : BB) {
      if ( auto *Ret = dyn_cast<ReturnInst>(&I)) {
        auto *KernelID = ConstantInt::get( IntegerType::getInt32Ty(M->getContext()),
                                           Kernel.getID());

        IRBuilder<> IRB(Ret);
        auto *Ptr = PointerType::get(GV->getValueType(), 0);
        auto *Cast = IRB.CreatePointerBitCastOrAddrSpaceCast(GV, Ptr);
        auto *GEP = IRB.CreateInBoundsGEP( GV->getValueType(), Cast, {
                                           ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0),
                                           ConstantInt::get(IntegerType::getInt64Ty(M->getContext()), 0)});
        if ( !IRB.CreateCall(Hook, {GEP, KernelID}))
          return false;
      }
    }
  }
  return true;
}

static bool initInstrumentation(Module &M, const std::vector<Kernel>& Kernels) {

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

  auto *Ty = ArrayType::get( IntegerType::getInt8Ty(M.getContext()), Kernels.size());

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

  if ( TraceStatusGV) {
    TraceStatusGV->setDSOLocal(true);
    TraceStatusGV->setAlignment(MaybeAlign(1));
  }

  return TraceStatusGV;
}

bool InstrumentPrintfPass::runOnModule(Module &M) {

  if ( M.getTargetTriple().find("nvptx") == std::string::npos)
    return false;

  if ( !getAnalysis<TypeCheckerPass>().moduleTypechecks())
    return false;

  InstrumentedFunctions.clear();
  GlobalVariableForSymbol.clear();

  if ( !isDeviceRTLinked(M)) {
#ifdef KERMA_OPT_PLUGIN
    llvm::report_fatal_error("KermaDeviceRT not found in " + M.getName());
#else
    LinkDeviceRTPass LinkKermaRTDevice;
    LinkKermaRTDevice.runOnModule(M);
#endif
  }

  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();

#ifdef KERMA_OPT_PLUGIN
  Mode = InstruMode.getValue();
  if ( !InstruTarget.getValue().empty()) {
    auto vals = parseDelimStr(InstruTarget, ',');
    for ( auto&& val : vals)
      Targets.push_back(val);
  }
  WithColor(errs(), HighlightColor::Note) << '[';
  WithColor(errs(), HighlightColor::String) << formatv("{0,15}", "Instrumenter");
  WithColor(errs(), HighlightColor::Note) << ']';
  errs() << " mode:" << Mode << ", #kernels:" << Kernels.size() << '\n';
#endif

  if ( !initInstrumentation(M, Kernels)) {
    WithColor::warning() << " Failed to init instrumentation. Exiting" << '\n';
    return false;
  }

  bool Changed = false;

  for ( auto &Kernel : Kernels) {

    bool KernelInstru = false;

    if ( auto *TraceStatus = instrumentTraceStatus(Kernel) ) {
      KernelInstru |= instrumentMeta(Kernel, TraceStatus);
      KernelInstru &= instrumentAccesses(Kernel, TraceStatus);
      KernelInstru &= instrumentStopTracing(Kernel);
    }

    if ( !KernelInstru) {
      WithColor::warning() << " Failed to instrument kernel: "<< Kernel.getDemangledName() << '\n';
    }

    Changed |= KernelInstru;
  }

  return Changed;
}

void InstrumentPrintfPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
  AU.addRequired<TypeCheckerPass>();
  AU.addRequired<CanonicalizerPass>();
}

Mode InstrumentPrintfPass::getMode() { return Mode; }

char InstrumentPrintfPass::ID = 4;

InstrumentPrintfPass::InstrumentPrintfPass(bool IgnoreLocal)
: IgnoreLocal(IgnoreLocal), ModulePass(ID) {}

InstrumentPrintfPass::InstrumentPrintfPass(const std::vector<std::string>& Targets, bool IgnoreLocal)
: IgnoreLocal(IgnoreLocal), ModulePass(ID) {
  for ( const auto& target : Targets)
    this->Targets.push_back(target);
}

bool InstrumentPrintfPass::hasTargetFunction() { return !this->Targets.empty(); }

} // namespace kerma

static RegisterPass<kerma::InstrumentPrintfPass> RegisterMemOpInstrumentationPass(
        /* arg      */ "kerma-instru",
        /* name     */ "Instrument memory operations in CUDA kernels",
        /* CFGOnly  */ false,
        /* analysis */ false);