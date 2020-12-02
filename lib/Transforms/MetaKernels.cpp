#include "kerma/Transforms/MetaKernels.h"
#include "kerma/Analysis/DetectAssumptions.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/MemoryAccessTree.h"
#include "kerma/Base/Kernel.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Transforms/Instrument/InstrumentPrintf.h"
#include "kerma/Transforms/Materializer.h"
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

namespace kerma {

using namespace llvm;

/**
https://stackoverflow.com/questions/14687367/how-to-insert-a-function-in-llvm-module
https://stackoverflow.com/questions/61189720/llvm-clone-function-pass-to-different-module
*/

static Function *findFunction(Module &M, const std::string &FName) {
  for (auto &F : M) {
    if (demangleFnWithoutArgs(F) == FName)
      return &F;
  }
  return nullptr;
}

static const Value *ReverseLookup(ValueToValueMapTy &VMap, Value *V) {
  for (auto E : VMap) {
    if (E.second == V)
      return E.first;
  }
  return nullptr;
}

static bool InstrumentG(Kernel &Kernel, Function *MetaKernel,
                        ValueToValueMapTy &VMap, Module &M) {
  Function *Record = findFunction(M, "record_g");
  SmallSet<Instruction *, 32> RemoveSet;
  SmallSet<CallInst *, 32> RemoveInstrinsics;
  for (auto &BB : *MetaKernel) {
    for (auto &I : BB) {
      if (isa<LoadInst>(&I) || isa<StoreInst>(&I))
        RemoveSet.insert(&I);
      else if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (nvvm::isBarrier(*CI->getCalledFunction()))
          RemoveSet.insert(CI);
        else if (nvvm::isMathFunction(*CI->getCalledFunction())) {
          auto NewF = getMathFunctionFor(*CI->getCalledFunction());
          auto *F = findFunction(M, NewF);
          if (!F) {
            WithColor::warning() << "Could not find x86 math function for "
                                 << CI->getCalledFunction()->getName() << '\n';
            return false;
          } else
            CI->setCalledFunction(F);
        }
      }
    }
  }
  auto *MAT = Kernel.getMAT();
  for (auto *I : RemoveSet) {
    GetElementPtrInst *GEP = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(I))
      GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
    else if (auto *SI = dyn_cast<StoreInst>(I))
      GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
    if (GEP) {
      if (auto *OrigV = ReverseLookup(VMap, I)) {
        if (auto *OrigI = dyn_cast<Instruction>(OrigV)) {
          if (auto *MA = MAT->getAccessForInst(OrigI)) {
            GEP->setOperand(
                0, Constant::getNullValue(GEP->getPointerOperandType()));
            IRBuilder<> IRB(I);
            Value *BidZ = MetaKernel->getArg(MetaKernel->arg_size() - 6);
            Value *BidY = MetaKernel->getArg(MetaKernel->arg_size() - 5);
            Value *BidX = MetaKernel->getArg(MetaKernel->arg_size() - 4);
            Value *TidZ = MetaKernel->getArg(MetaKernel->arg_size() - 3);
            Value *TidY = MetaKernel->getArg(MetaKernel->arg_size() - 2);
            Value *TidX = MetaKernel->getArg(MetaKernel->arg_size() - 1);
            Value *AccessID = ConstantInt::get(IRB.getInt32Ty(), MA->getID());
            Value *Offset = IRB.CreatePtrToInt(GEP, IRB.getInt64Ty());
            auto *CallRecord = CallInst::Create(
                Record, {BidZ, BidY, BidX, TidZ, TidY, TidX, AccessID, Offset},
                "", I);
            I->replaceAllUsesWith(UndefValue::get(GEP->getResultElementType()));
          }
        }
      }
    }
  }

  for (auto *I : RemoveSet)
    I->eraseFromParent();
  for (auto *CI : RemoveInstrinsics)
    CI->eraseFromParent();

  return true;
}

static bool Instrument(Kernel &Kernel, Function *MetaKernel,
                       ValueToValueMapTy &VMap, Module &M) {
  Function *Record = findFunction(M, "record");
  SmallSet<Instruction *, 32> RemoveSet;
  SmallSet<CallInst *, 32> RemoveInstrinsics;

  for (auto &BB : *MetaKernel) {
    for (auto &I : BB) {
      if (isa<LoadInst>(&I) || isa<StoreInst>(&I))
        RemoveSet.insert(&I);
      else if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (nvvm::isBarrier(*CI->getCalledFunction()))
          RemoveSet.insert(CI);
        else if (nvvm::isMathFunction(*CI->getCalledFunction())) {
          auto NewF = getMathFunctionFor(*CI->getCalledFunction());
          auto *F = findFunction(M, NewF);
          if (!F) {
            WithColor::warning() << "Could not find x86 math function for "
                                 << CI->getCalledFunction()->getName() << '\n';
            return false;
          } else
            CI->setCalledFunction(F);
        }
      }
    }
  }

  auto *MAT = Kernel.getMAT();

  for (auto *I : RemoveSet) {
    GetElementPtrInst *GEP = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(I))
      GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
    else if (auto *SI = dyn_cast<StoreInst>(I))
      GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
    if (GEP) {
      if (auto *OrigV = ReverseLookup(VMap, I)) {
        if (auto *OrigI = dyn_cast<Instruction>(OrigV)) {
          if (auto *MA = MAT->getAccessForInst(OrigI)) {
            GEP->setOperand(
                0, Constant::getNullValue(GEP->getPointerOperandType()));
            IRBuilder<> IRB(I);
            Value *TidZ = MetaKernel->getArg(MetaKernel->arg_size() - 3);
            Value *TidY = MetaKernel->getArg(MetaKernel->arg_size() - 2);
            Value *TidX = MetaKernel->getArg(MetaKernel->arg_size() - 1);
            Value *AccessID = ConstantInt::get(IRB.getInt32Ty(), MA->getID());
            Value *Offset = IRB.CreatePtrToInt(GEP, IRB.getInt64Ty());
            auto *CallRecord = CallInst::Create(
                Record, {TidZ, TidY, TidX, AccessID, Offset}, "", I);
            I->replaceAllUsesWith(UndefValue::get(GEP->getResultElementType()));
          }
        }
      }
    }
  }

  for (auto *I : RemoveSet)
    I->eraseFromParent();
  for (auto *CI : RemoveInstrinsics)
    CI->eraseFromParent();

  return true;
}

static Function *CreateGMetaKernelFor(Kernel &K, AssumptionInfo &AI,
                                      Module &MetaDriverModule) {
  Type *Result = Type::getVoidTy(K.getFunction()->getContext());
  std::vector<Type *> Params;
  for (auto &Arg : K.getFunction()->args())
    Params.push_back(Arg.getType());

  // Add 6 extra arguments
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));

  FunctionType *Ty = FunctionType::get(Result, Params, false);
  auto *MetaKernel =
      Function::Create(Ty, GlobalValue::PrivateLinkage,
                       "_meta_g_" + K.getName(), MetaDriverModule);

  ValueToValueMapTy VMap;
  ClonedCodeInfo CCI;
  SmallVector<ReturnInst *, 8> Returns;
  // Clone the original kernel into a new function
  CloneFunctionInto(MetaKernel, K.getFunction(), VMap, false, Returns);

  // Give proper names to the clone's arguments
  for (size_t i = 0; i < K.getFunction()->arg_size(); ++i)
    MetaKernel->getArg(i)->setName(K.getFunction()->getArg(i)->getName());
  MetaKernel->getArg(K.getFunction()->arg_size())->setName("bidz");
  MetaKernel->getArg(K.getFunction()->arg_size() + 1)->setName("bidy");
  MetaKernel->getArg(K.getFunction()->arg_size() + 2)->setName("bidx");
  MetaKernel->getArg(K.getFunction()->arg_size() + 3)->setName("tidz");
  MetaKernel->getArg(K.getFunction()->arg_size() + 4)->setName("tidy");
  MetaKernel->getArg(K.getFunction()->arg_size() + 5)->setName("tidx");

  // fix the clone's attributes
  MetaKernel->setAttributes(
      findFunction(MetaDriverModule, "record")->getAttributes());
  MetaKernel->removeFnAttr(Attribute::OptimizeNone);

  // replace calls to blockIdx.x etc with the appropriate values
  // the call instructions now become dead code and will be DCEd
  MaterializeBlockIdx(MetaKernel,
                      MetaKernel->getArg(MetaKernel->arg_size() - 6),
                      MetaKernel->getArg(MetaKernel->arg_size() - 5),
                      MetaKernel->getArg(MetaKernel->arg_size() - 4));
  MaterializeThreadIdx(MetaKernel,
                       MetaKernel->getArg(MetaKernel->arg_size() - 3),
                       MetaKernel->getArg(MetaKernel->arg_size() - 2),
                       MetaKernel->getArg(MetaKernel->arg_size() - 1));

  // Instrument the meta
  if (!InstrumentG(K, MetaKernel, VMap, MetaDriverModule))
    return nullptr;
  return MetaKernel;
}

static Function *CreateMetaKernelFor(Kernel &K, const Index &Bid,
                                     AssumptionInfo &AI,
                                     Module &MetaDriverModule) {
  Type *Result = Type::getVoidTy(K.getFunction()->getContext());
  std::vector<Type *> Params;
  for (auto &Arg : K.getFunction()->args())
    Params.push_back(Arg.getType());

  // Add 3 extra arguments for tid.{z,y,x}
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));
  Params.push_back(Type::getInt32Ty(K.getFunction()->getContext()));

  FunctionType *Ty = FunctionType::get(Result, Params, false);
  auto *MetaKernel = Function::Create(Ty, GlobalValue::PrivateLinkage,
                                      "_meta_" + K.getName(), MetaDriverModule);

  ValueToValueMapTy VMap;
  ClonedCodeInfo CCI;
  SmallVector<ReturnInst *, 8> Returns;
  // Clone the original kernel into a new function
  CloneFunctionInto(MetaKernel, K.getFunction(), VMap, false, Returns);

  // Give proper names to the clone's arguments
  for (size_t i = 0; i < K.getFunction()->arg_size(); ++i)
    MetaKernel->getArg(i)->setName(K.getFunction()->getArg(i)->getName());
  MetaKernel->getArg(K.getFunction()->arg_size())->setName("tidz");
  MetaKernel->getArg(K.getFunction()->arg_size() + 1)->setName("tidy");
  MetaKernel->getArg(K.getFunction()->arg_size() + 2)->setName("tidx");

  // fix the clone's attributes
  MetaKernel->setAttributes(
      findFunction(MetaDriverModule, "record")->getAttributes());
  MetaKernel->removeFnAttr(Attribute::OptimizeNone);

  // replace calls to blockIdx.x etc with the appropriate values
  // the call instructions now become dead code and will be DCEd
  MaterializeBlockIdx(MetaKernel, Bid);
  MaterializeThreadIdx(MetaKernel,
                       MetaKernel->getArg(MetaKernel->arg_size() - 3),
                       MetaKernel->getArg(MetaKernel->arg_size() - 2),
                       MetaKernel->getArg(MetaKernel->arg_size() - 1));

  // Instrument the meta
  if (!Instrument(K, MetaKernel, VMap, MetaDriverModule))
    return nullptr;
  return MetaKernel;
}

static Instruction *CreateMetaKernelCall(Function *MetaKernel, Kernel &K,
                                         AssumptionInfo &AI, Value *TidZ,
                                         Value *TidY, Value *TidX) {
  auto &Ctx = K.getFunction()->getContext();
  std::vector<Value *> Args;
  for (size_t i = 0; i < MetaKernel->arg_size(); ++i) {
    if (i == K.getFunction()->arg_size()) { // threadIdx.z
      Args.push_back(TidZ);
    } else if (i == K.getFunction()->arg_size() + 1) { // threadIdx.y
      Args.push_back(TidY);
    } else if (i == K.getFunction()->arg_size() + 2) { // threadIdx.x
      Args.push_back(TidX);
    } else {
      auto *OriginalArg = K.getFunction()->getArg(i);
      if (OriginalArg->getType()->isPointerTy()) { // NULL for all pointer args
        Args.push_back(
            Constant::getNullValue(MetaKernel->getArg(i)->getType()));
      } else if (auto *A = AI.getForArg(OriginalArg)) {
        // we found an assumption for a scalar arg
        if (auto *FPA = dyn_cast<FPAssumption>(A))
          Args.push_back(ConstantFP::get(Ctx, APFloat(FPA->getValue())));
        else if (auto *IA = dyn_cast<IAssumption>(A))
          Args.push_back(ConstantInt::get(
              dyn_cast<IntegerType>(OriginalArg->getType()), IA->getValue()));
        else
          WithColor::warning() << "MetaKernelFullPass: Invalid assumption " << A
                               << " for val argument at pos " << i << '\n';
      } else {
        WithColor::warning() << "MetaKernelFullPass: No known value "
                                "to use for argument "
                             << *OriginalArg << '\n';
      }
    }
  }

  CallInst *CallMetaKernel = CallInst::Create(MetaKernel, Args);
  return CallMetaKernel;
}

static Instruction *CreateGMetaKernelCall(Function *MetaKernel, Kernel &K,
                                          AssumptionInfo &AI, Value *BidZ,
                                          Value *BidY, Value *BidX, Value *TidZ,
                                          Value *TidY, Value *TidX) {
  auto &Ctx = K.getFunction()->getContext();
  std::vector<Value *> Args;
  for (size_t i = 0; i < MetaKernel->arg_size(); ++i) {
    if (i == K.getFunction()->arg_size()) { // blockIdx.z
      Args.push_back(BidZ);
    } else if (i == K.getFunction()->arg_size() + 1) { // blockIdx.y
      Args.push_back(BidY);
    } else if (i == K.getFunction()->arg_size() + 2) { // blockIdx.x
      Args.push_back(BidX);
    } else if (i == K.getFunction()->arg_size() + 3) { // threadIdx.z
      Args.push_back(TidZ);
    } else if (i == K.getFunction()->arg_size() + 4) { // threadIdx.y
      Args.push_back(TidY);
    } else if (i == K.getFunction()->arg_size() + 5) { // threadIdx.x
      Args.push_back(TidX);
    } else {
      auto *OriginalArg = K.getFunction()->getArg(i);
      if (OriginalArg->getType()->isPointerTy()) { // NULL for all pointer args
        Args.push_back(
            Constant::getNullValue(MetaKernel->getArg(i)->getType()));
      } else if (auto *A = AI.getForArg(OriginalArg)) {
        // we found an assumption for a scalar arg
        if (auto *FPA = dyn_cast<FPAssumption>(A))
          Args.push_back(ConstantFP::get(Ctx, APFloat(FPA->getValue())));
        else if (auto *IA = dyn_cast<IAssumption>(A))
          Args.push_back(ConstantInt::get(
              dyn_cast<IntegerType>(OriginalArg->getType()), IA->getValue()));
        else
          WithColor::warning() << "MetaKernelFullPass: Invalid assumption " << A
                               << " for val argument at pos " << i << '\n';
      } else {
        WithColor::warning() << "MetaKernelFullPass: No known value "
                                "to use for argument "
                             << *OriginalArg << '\n';
      }
    }
  }
  CallInst *CallMetaKernel = CallInst::Create(MetaKernel, Args);
  return CallMetaKernel;
}

static void Prepare(Function *Main, Function *Record, Function *BlockLoop) {
  BlockLoop->removeFnAttr(Attribute::OptimizeNone);
}

static Instruction *ThreadMode(Function *MetaKernel, Kernel &Kernel,
                               AssumptionInfo &AI, Index &Tid,
                               Instruction *InsertPt) {
  auto &Ctx = MetaKernel->getContext();
  auto *Call = CreateMetaKernelCall(
      MetaKernel, Kernel, AI,
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Tid.z),
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Tid.y),
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Tid.x));
  // Call->insertAfter(InsertPt);
  if (InsertPt->isTerminator())
    Call->insertBefore(InsertPt);
  else
    Call->insertAfter(InsertPt);
  return Call;
}

static Instruction *GridMode(Function *MetaKernel, Function *Wrapper,
                             Kernel &Kernel, AssumptionInfo &AI,
                             ScalarEvolution &SCEV, LoopInfo &LI,
                             Instruction *InsertPt) {
  auto &Ctx = MetaKernel->getContext();
  auto &LoopBZ = *LI.begin();
  auto &LoopBY = *LoopBZ->getSubLoops().begin();
  auto &LoopBX = *LoopBY->getSubLoops().begin();
  auto &LoopZ = *LoopBX->getSubLoops().begin();
  auto *LoopY = *LoopZ->getSubLoops().begin();
  auto *LoopX = *LoopY->getSubLoops().begin();

  auto *IVBZ = LoopBZ->getCanonicalInductionVariable();
  auto *IVBY = LoopBY->getCanonicalInductionVariable();
  auto *IVBX = LoopBX->getCanonicalInductionVariable();
  auto *IVTZ = LoopZ->getCanonicalInductionVariable();
  auto *IVTY = LoopY->getCanonicalInductionVariable();
  auto *IVTX = LoopX->getCanonicalInductionVariable();

  auto *HeaderTerminator =
      dyn_cast<BranchInst>(LoopX->getHeader()->getTerminator());
  auto *Body = HeaderTerminator->getSuccessor(0);
  auto *MetaKernelCall = CreateGMetaKernelCall(MetaKernel, Kernel, AI, IVBZ,
                                               IVBY, IVBX, IVTZ, IVTY, IVTX);
  MetaKernelCall->insertBefore(Body->getFirstNonPHI());

  std::vector<Value *> WrapperArgs;
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getGrid().z));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getGrid().y));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getGrid().x));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().z));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().y));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().x));
  auto *WrapperCall = CallInst::Create(Wrapper, WrapperArgs);
  if (InsertPt->isTerminator())
    WrapperCall->insertBefore(InsertPt);
  else
    WrapperCall->insertAfter(InsertPt);
  return WrapperCall;
}

static Instruction *BlockMode(Function *MetaKernel, Function *Wrapper,
                              Kernel &Kernel, AssumptionInfo &AI,
                              ScalarEvolution &SCEV, LoopInfo &LI,
                              Instruction *InsertPt) {

  // https://groups.google.com/g/llvm-dev/c/YfQRheMqMkM/m/Abl1DIWcAQAJ

  auto &Ctx = MetaKernel->getContext();
  auto *LoopZ = *LI.begin();
  auto *LoopY = *LoopZ->getSubLoops().begin();
  auto *LoopX = *LoopY->getSubLoops().begin();

  // The following calls are fine because we know the loops
  // of the wrapper start at 0. For the general case we would
  // need to run the LoopRotate pass for getInductionVariable()
  // to return non-null. Thus, this is an optimation
  auto *IVZ = LoopZ->getCanonicalInductionVariable();
  auto *IVY = LoopY->getCanonicalInductionVariable();
  auto *IVX = LoopX->getCanonicalInductionVariable();

  // auto *MetaKernelCall =
  //     CreateMetaKernelCall(MetaKernel, Kernel, AI, IVZ, IVY, IVX);
  // MetaKernelCall->insertBefore(LoopX->getBlocks().front()->getFirstNonPHI());

  auto *HeaderTerminator =
      dyn_cast<BranchInst>(LoopX->getHeader()->getTerminator());
  auto *Body = HeaderTerminator->getSuccessor(0);
  auto *MetaKernelCall =
      CreateMetaKernelCall(MetaKernel, Kernel, AI, IVZ, IVY, IVX);
  MetaKernelCall->insertBefore(Body->getFirstNonPHI());

  std::vector<Value *> WrapperArgs;
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().z));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().y));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                                         AI.getLaunch(Kernel)->getBlock().x));

  auto *WrapperCall = CallInst::Create(Wrapper, WrapperArgs);
  if (InsertPt->isTerminator())
    WrapperCall->insertBefore(InsertPt);
  else
    WrapperCall->insertAfter(InsertPt);
  return WrapperCall;
}

static Instruction *WarpMode(Function *MetaKernel, Function *Wrapper,
                             Kernel &Kernel, AssumptionInfo &AI,
                             unsigned WarpIdx, ScalarEvolution &SCEV,
                             LoopInfo &LI, Instruction *InsertPt) {
  auto &Ctx = MetaKernel->getContext();
  auto *LoopZ = *LI.begin();
  auto *LoopY = *LoopZ->getSubLoops().begin();
  auto *LoopX = *LoopY->getSubLoops().begin();
  auto &IVZ = *LoopZ->getHeader()->phis().begin();
  auto &IVY = *LoopY->getHeader()->phis().begin();
  auto &IVX = *LoopX->getHeader()->phis().begin();

  auto *HeaderTerminator =
      dyn_cast<BranchInst>(LoopX->getHeader()->getTerminator());
  auto *Body = HeaderTerminator->getSuccessor(0);
  auto *MetaKernelCall =
      CreateMetaKernelCall(MetaKernel, Kernel, AI, &IVZ, &IVY, &IVX);
  MetaKernelCall->insertBefore(Body->getFirstNonPHI());

  unsigned Lane0 = WarpIdx * 32;
  auto &BDim = AI.getLaunch(Kernel)->getBlock();
  auto Lane03D = Index::delinearize(Lane0, BDim);
  errs() << Lane03D << '\n';

  std::vector<Value *> WrapperArgs;
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx), BDim.z));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx), BDim.y));
  WrapperArgs.push_back(ConstantInt::get(IntegerType::getInt32Ty(Ctx), BDim.x));
  WrapperArgs.push_back(
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Lane03D.z));
  WrapperArgs.push_back(
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Lane03D.y));
  WrapperArgs.push_back(
      ConstantInt::get(IntegerType::getInt32Ty(Ctx), Lane03D.x));

  auto *WrapperCall = CallInst::Create(Wrapper, WrapperArgs);
  if (InsertPt->isTerminator())
    WrapperCall->insertBefore(InsertPt);
  else
    WrapperCall->insertAfter(InsertPt);
  return WrapperCall;
}

static Instruction *RecordKernelMetadata(Kernel &Kernel, Function *MetadataFun,
                                         Mode Mode, Instruction *InsertPt) {
  auto &Ctx = MetadataFun->getContext();
  auto *Call = CallInst::Create(
      MetadataFun,
      {
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getID()), // kernel id
          ConstantInt::get(IntegerType::getInt8Ty(Ctx), Mode),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getGrid().z),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getGrid().y),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getGrid().x),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getBlock().z),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getBlock().y),
          ConstantInt::get(IntegerType::getInt32Ty(Ctx),
                           Kernel.getLaunchAssumption()->getBlock().x),
      });
  if (InsertPt->isTerminator())
    Call->insertBefore(InsertPt);
  else
    Call->insertAfter(InsertPt);
  return Call;
}

bool MetaKernelFullPass::runOnModule(llvm::Module &M) {
  WithColor::note() << "MetaKernelFullPass: Running in '" << ModeStr(Mode)
                    << "' mode\n";
  if (KI.getKernels().empty()) {
    WithColor::warning() << "MetaKernelFullPass: KernelInfo object "
                            "empty. Nothing to do!"
                         << '\n';
    return false;
  }

  auto *DeviceModule = KI.getKernels().begin()->getFunction()->getParent();
  auto &Ctx = M.getContext();

  Function *Main = findFunction(M, "main");
  Function *Meta = findFunction(M, "kernel_meta");
  Function *Record = findFunction(M, "record");
  Function *GridLoop = findFunction(M, "grid_loop");
  Function *BlockLoop = findFunction(M, "block_loop");
  Function *WarpLoop = findFunction(M, "warp_loop");

  Prepare(Main, Record, BlockLoop);

  auto *InsertPt = Main->begin()->getFirstNonPHI();

  for (auto &Kernel : KI.getKernels()) {
    Function *MetaKernel =
        (Mode == GRID) ? CreateGMetaKernelFor(Kernel, AI, M)
                       : CreateMetaKernelFor(Kernel, TargetBlock, AI, M);

    if (!MetaKernel) {
      WithColor::error()
          << "MetaKernelFullPass: Failed to create meta kernel stub for "
          << Kernel.getName() << '\n';
      break;
    }

    InsertPt = RecordKernelMetadata(Kernel, Meta, Mode, InsertPt);

    if (Mode == THREAD) {

      InsertPt = ThreadMode(MetaKernel, Kernel, AI, TargetThread, InsertPt);

    } else if (Mode == GRID) {

      ValueToValueMapTy VMap;
      auto *Wrapper = CloneFunction(GridLoop, VMap);
      Wrapper->setName("_meta_grid_" + Kernel.getName());

      auto &SCEV = getAnalysis<ScalarEvolutionWrapperPass>(*Wrapper).getSE();
      auto &LI = getAnalysis<LoopInfoWrapperPass>(*Wrapper).getLoopInfo();
      InsertPt = GridMode(MetaKernel, Wrapper, Kernel, AI, SCEV, LI, InsertPt);

    } else if (Mode == BLOCK) {

      ValueToValueMapTy VMap;
      auto *Wrapper = CloneFunction(BlockLoop, VMap);
      Wrapper->setName("_meta_block_" + Kernel.getName());

      auto &SCEV = getAnalysis<ScalarEvolutionWrapperPass>(*Wrapper).getSE();
      auto &LI = getAnalysis<LoopInfoWrapperPass>(*Wrapper).getLoopInfo();
      InsertPt = BlockMode(MetaKernel, Wrapper, Kernel, AI, SCEV, LI, InsertPt);

    } else {

      ValueToValueMapTy VMap;
      auto *Wrapper = CloneFunction(WarpLoop, VMap);
      Wrapper->setName("_meta_warp_" + Kernel.getName());

      auto &SCEV = getAnalysis<ScalarEvolutionWrapperPass>(*Wrapper).getSE();
      auto &LI = getAnalysis<LoopInfoWrapperPass>(*Wrapper).getLoopInfo();
      InsertPt = WarpMode(MetaKernel, Wrapper, Kernel, AI, TargetWarp, SCEV, LI,
                          InsertPt);
    }
  }

  return true;
}

void MetaKernelFullPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
}

char MetaKernelFullPass::ID = 48;

MetaKernelFullPass::MetaKernelFullPass(KernelInfo &KI, AssumptionInfo &AI,
                                       enum Mode M, const Index &TargetBlock,
                                       const Index &TargetThread,
                                       unsigned TargetWarp)
    : ModulePass(ID), KI(KI), AI(AI), Mode(M), TargetBlock(TargetBlock),
      TargetThread(TargetThread), TargetWarp(TargetWarp) {}

} // namespace kerma