#include "kerma/Transforms/Instrument/InstruMemOpPrintf.h"

#include "kerma/Analysis/DetectKernels.h"
#include "kerma/Analysis/InferDimensions.h"
#include "kerma/Analysis/InferAddressSpaces.h"
#include "kerma/Analysis/Symbolize.h"
#include "kerma/Analysis/Typecheck.h"
#include "kerma/NVVM/NVVM.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/RT/Util.h"
#include "kerma/Support/Demangle.h"
#include "kerma/Support/Parse.h"
#include "kerma/Utils/LLVMMetadata.h"
#include "kerma/Transforms/Instrument/LinkDeviceRT.h"
#include "kerma/Transforms/Canonicalize/Canonicalizer.h"

#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include <algorithm>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Scalar.h>
#include <memory>
#include <string>
#include <utility>

#include <llvm/Analysis/MemorySSA.h>

using namespace llvm;

#define TAB "       "

#ifdef KERMA_OPT_PLUGIN

#include "llvm/Support/CommandLine.h"


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



char InstruMemOpPrintfPass::ID = 4;

InstruMemOpPrintfPass::InstruMemOpPrintfPass(MemOp Op, bool IgnoreLocal)
: TargetOp(Op), IgnoreLocal(IgnoreLocal), ModulePass(ID) {}

InstruMemOpPrintfPass::InstruMemOpPrintfPass(const std::vector<std::string>& Targets, bool IgnoreLocal)
: InstruMemOpPrintfPass::InstruMemOpPrintfPass(Targets, MemOp::All, IgnoreLocal)
{}

InstruMemOpPrintfPass::InstruMemOpPrintfPass(const std::vector<std::string>& Targets,
                                             MemOp Op, bool IgnoreLocal)
: TargetOp(Op), IgnoreLocal(IgnoreLocal), ModulePass(ID)
{
  for ( const auto& target : Targets)
    this->Targets.push_back(target);
}

bool InstruMemOpPrintfPass::hasTargetFunction() { return !this->Targets.empty(); }

MemOp InstruMemOpPrintfPass::getTargetOp() { return TargetOp; }

// bool InstruMemOpPrintfPass::isInstrumented(Function &F) {
//   return this->InstrumentedFunctions.find(&F) != this->InstrumentedFunctions.end();
// }

GlobalVariable *insertGlobalStr(Module &M, llvm::StringRef Str) {
  static unsigned int counter = 0;

  auto* CharTy = IntegerType::get(M.getContext(), 8);

  std::vector<Constant*> chars(Str.size());
  for ( unsigned int i = 0; i < Str.size(); ++i)
    chars[i] = ConstantInt::get(CharTy, Str[i]);
  chars.push_back( ConstantInt::get(CharTy, 0));

  auto* StrTy = ArrayType::get(CharTy, chars.size());

  auto *G = M.getOrInsertGlobal(std::string("arr") + std::to_string(counter++), StrTy);

  if ( G) {
    if ( auto* GV = dyn_cast<GlobalVariable>(G)) {
      GV->setInitializer(ConstantArray::get(StrTy, chars));
      GV->setConstant(true);
      GV->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
      GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      return GV;
    }
  }
  return nullptr;
}

static bool GEPHasAddressSpaceCast(GetElementPtrInst *GEP) {
  if ( auto *CE = dyn_cast<ConstantExpr>(GEP->getOperand(0)))
    if ( auto *OP = dyn_cast<Operator>(CE))
      if ( OP->getOpcode() == Instruction::AddrSpaceCast)
        return true;
  return false;
}

static StringRef findSymbolForGlobal(GlobalVariable *GV) {
  static int id = 0;

  // llvm::errs() << "findSymbolForGlobal: " << *GV << "\n";
  StringRef name;
  if ( auto *DIGVExpr = dyn_cast<DIGlobalVariableExpression>(GV->getMetadata("dbg")))
    name = DIGVExpr->getVariable()->getName();
  else if ( auto *DIGV = dyn_cast<DIGlobalVariable>(GV->getMetadata("dbg")))
    name = DIGV->getName();

  if ( name.empty())
    name = GV->getName();

  if ( name.empty())
    name = StringRef(std::string("gvar") + std::to_string(id++));

  GV->setName(name);

  return name;
}

// DILocalVariable *findMDForArgument(Argument *Arg) {
//   for ( auto &BB : *Arg->getParent() ) {
//     for ( auto &I : BB) {
//       if ( auto *DVI = dyn_cast<DbgValueInst>(&I)) {
//         if ( auto *op1 = dyn_cast<MetadataAsValue>(DVI->getOperand(1))) {
//           if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
//             if ( DILV->getArg() - 1 == Arg->getArgNo())
//               return DILV;
//           }
//         }
//       }
//       else if ( auto* DDI = dyn_cast<DbgDeclareInst>(&I)) {
//         if ( auto *op1 = dyn_cast<MetadataAsValue>(DDI->getOperand(1)))
//           if ( auto *DILV = dyn_cast<DILocalVariable>(op1->getMetadata())) {
//             if ( DILV->getArg() - 1 == Arg->getArgNo())
//               return DILV;
//           }
//       }
//     }
//   }
//   return nullptr;
// }


StringRef getSymbolFromGEP(GetElementPtrInst *GEP) {
  auto Ptr = GEP->getOperand(0);

  if ( auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if ( auto *OP = dyn_cast<Operator>(Ptr)) {
      if ( OP->getOpcode() == Instruction::AddrSpaceCast) {
        auto *op1 = OP->getOperand(0);
        if ( auto* GV = dyn_cast<GlobalVariable>(op1))
          return findSymbolForGlobal(GV);
      }
    }
  }
  if ( auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
    return findSymbolForGlobal(GV);
  }
  if ( auto *Arg = dyn_cast<Argument>(Ptr))
    return findMDForArgument(Arg)->getName();
  return "symbol";
}

StringRef findOrCreateSymbolForArg(Argument *Arg) {
  static int argId = 0;
  if ( !Arg->getName().empty())
    return Arg->getName();
  if ( auto* MD = findMDForArgument(Arg)) {
    if ( !MD->getName().empty()) {
      Arg->setName(MD->getName());
      return MD->getName();
    }
  }
  Arg->setName(std::string("_") + std::to_string(argId++) + "arg_" + std::to_string(Arg->getArgNo()));
  return Arg->getName();
}


const std::string InstruMemOpPrintfPass::getOpString(MemOp op) {
  if ( op == Load)
    return "Load";
  else if ( op == Store)
    return "Store";
  else if ( op == Atomic)
    return "Atomic";
  else
    return "All";
}

const char InstruMemOpPrintfPass::getOpChar(MemOp op) {
  if ( op == Load)
    return 'L';
  else if ( op == Store)
    return 'S';
  else if ( op == Atomic)
    return 'A';
  else
    return '*';
}

static Value* stripCasts(Value *V) {
  auto *Res = V;
  while ( auto *cast = dyn_cast<BitCastOperator>(Res))
    Res = cast->getOperand(0);
  return Res;
}

  // https://stackoverflow.com/questions/58596205/how-to-generate-an-inline-getelementptr-instruction
  // https://github.com/spurious/safecode-mirror/blob/master/lib/ArrayBoundChecks/BreakConstantGEPs.cpp
  // return recordGEPAccess(GEP, &LI);
  // https://llvm.org/doxygen/NVPTXInferAddressSpaces_8cpp_source.html




/// Check if a LoadInst loads an aggregate argument.
/// For example the following:
///
///    __global__ void kernel(struct S MyStruct)
///       ...
///       int x = MyStruct.x
///       ...
///    }
///
/// Produces this IR:
///      %7 = getelementptr inbounds %struct.s, %struct.s* %4, i32 0, i32 1
///      %8 = load i32*, i32** %7, align 8
///
bool isLoadForAggregateArg(Instruction *I) {
  if ( auto *LI = dyn_cast<LoadInst>(I)) {
    auto* Ptr = LI->getPointerOperand();
    if ( auto *Arg = dyn_cast<Argument>(Ptr)) {
      if ( Arg->getType()->isAggregateType())
        return true;
    } else if ( auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      if ( auto *Arg = dyn_cast<Argument>(GEP->getPointerOperand())) {
        if ( Arg->getType()->isAggregateType() ||
             Arg->getType()->isPointerTy() && Arg->getType()->getPointerElementType()->isAggregateType())
          return true;
      }
    }
  }
  return false;
}


/// Checks if a Load should be instrumented
/// Some Loads are loading intermediate pointers 
/// for GEPs We want to skip those.
/// Example:
///   arg: int A[10][10]
///   ...
///   int x = A[]
bool isInterestingLoad(LoadInst& LI) {
  auto* PtrOp = LI.getPointerOperand();
  auto* PtrTy = LI.getPointerOperandType();
  if ( isa<PointerType>(PtrTy->getPointerElementType()))
    // case 1: struct arguments
    // case 2: actually loading a pointer
    return false;
  return true;
}




/// Check if a function is a device atomic
static bool isAtomic(Function &F) { return nvvm::isAtomic(demangle(F.getName())); }

Mode InstruMemOpPrintfPass::getMode() { return Mode; }


SmallSet<GlobalVariable *, 32> getGlobalsUsedInKernel(Function &Kernel) {
  SmallSet<GlobalVariable *, 32> Globals;
  for ( auto& G : Kernel.getParent()->globals()) {
    if ( G.getName().startswith("__kerma"))
      continue;
    for ( auto *User : G.users()) {
      if ( auto *CE = dyn_cast<ConstantExpr>(User)) {
        bool done = false;
        // go through the users of the constant expr
        for ( auto *CEUSer : CE->users())
          // if its used in an instruction
          if ( auto *I = dyn_cast<Instruction>(CEUSer)) {
            // check if the instruction is part of our target function
            if ( I->getParent()->getParent() == &Kernel) {
              Globals.insert(&G);
              done = true;
              break;
            }
          }
        if (done) break;
      }
      else if ( auto *I = dyn_cast<Instruction>(User)) {
        if ( I->getParent()->getParent() == &Kernel)
          Globals.insert(&G);
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

bool InstruMemOpPrintfPass::instrumentGlobalBaseAddresses(Function &Kernel, unsigned char KernelId, Function &Hook) {
  auto GVs = getGlobalsUsedInKernel(Kernel);

  IRBuilder<> IRB(Kernel.front().getFirstNonPHI());

  bool changes = false;
  for ( auto *GV : GVs) {
    auto name = findSymbolForGlobal(GV);
    if ( GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] = IRB.CreateGlobalStringPtr(name, "__kerma_sym");

    // IR globals are always pointers, get type of the pointee
    // if the type of the pointe is an array then get the type
    // of the elements of the array
    auto *GVPointeeTy = getGlobalElementTy(GV->getValueType());

    if ( !GVPointeeTy) {
      llvm::errs() << "[instru] Error recording base address for global: " << *GV << '\n';
      continue;
    }

    // create a pointer to the element type
    auto *GVPointeeTyPtr = PointerType::get(GVPointeeTy, 0);
    auto *Cast = IRB.CreateAddrSpaceCast(GV, GVPointeeTyPtr);
    auto PtrToStore = IRB.CreatePtrToInt(Cast,  IRB.getInt64Ty());
    ArrayRef<Value*> Args({ /* kernelid*/ ConstantInt::get(IRB.getInt8Ty(), KernelId),
                            /* symbol  */ GlobalVariableForSymbol[name],
                            /* addrspc */ ConstantInt::get(IRB.getInt8Ty(), GV->getAddressSpace()),
                            /* base    */ PtrToStore});
    auto *CI = IRB.CreateCall(&Hook, Args);
    CI->setCallingConv(CallingConv::PTX_Device);
    changes = true;
  }

  return changes;
}

bool InstruMemOpPrintfPass::instrumentArgBaseAddresses(Function &Kernel, unsigned char KernelId, Function &Hook) {
  IRBuilder<> IRB(Kernel.front().getFirstNonPHI());
  bool changes = false;
  for ( auto& Arg : Kernel.args()) {

    if ( Arg.hasAttribute(Attribute::ByVal))
      continue;

    auto ArgTy = Arg.getType();

    std::string name = findOrCreateSymbolForArg(&Arg);

    if ( GlobalVariableForSymbol.find(name) == GlobalVariableForSymbol.end())
      GlobalVariableForSymbol[name] = IRB.CreateGlobalStringPtr(name, "__kerma_sym");

    if ( auto *ArgPtrTy = dyn_cast<PointerType>(ArgTy)) {
      auto PtrToStore = IRB.CreatePtrToInt(&Arg, IRB.getInt64Ty());
      ArrayRef<Value*> Args({ /* kernelid*/ ConstantInt::get(IRB.getInt8Ty(), KernelId),
                              /* symbol  */ GlobalVariableForSymbol[name],
                              /* addrspc */ ConstantInt::get(IRB.getInt8Ty(), 1),
                              /* base    */ PtrToStore});
      IRB.CreateCall(&Hook, Args);
      changes = true;
    }
  }
  return changes;
}

bool InstruMemOpPrintfPass::instrumentBaseAddresses(Function &Kernel, unsigned char KernelId) {
  if ( auto *Hook = Kernel.getParent()->getFunction("__kerma_rec_base") ) {
    bool globalChanges = instrumentGlobalBaseAddresses(Kernel, KernelId, *Hook);
    bool argChanges = instrumentArgBaseAddresses(Kernel, KernelId, *Hook);
    return globalChanges || argChanges;
  }
  return false;
}

bool InstruMemOpPrintfPass::instrumentKernelMeta(llvm::Function &F, unsigned char id) {
  auto KernelName = demangleFnWithoutArgs(F);
  IRBuilder<> IRB(F.front().getFirstNonPHI());
  if ( GlobalVariableForSymbol.find(KernelName) == GlobalVariableForSymbol.end())
    GlobalVariableForSymbol[KernelName] = IRB.CreateGlobalStringPtr(KernelName, "__kerma_sym");

  auto *TraceFun = F.getParent()->getFunction("__kerma_rec_kernel");

  if ( !TraceFun) return false;

  ArrayRef<Value*> Args({/* id   */ ConstantInt::get(IRB.getInt8Ty(), id),
                         /* name */ GlobalVariableForSymbol[KernelName]});
  IRB.CreateCall(TraceFun, Args);
  // return true;

  return instrumentBaseAddresses(F, id);
}

bool InstruMemOpPrintfPass::instrumentAccess( llvm::Module *M, unsigned char KernelId,
    llvm::Value *Ptr, SourceLoc& Loc, llvm::Instruction *InsertBefore, MemOp op) {

  std::string symbol = "sym";
  IRBuilder<> IRB(InsertBefore);
  if ( GlobalVariableForSymbol.find(symbol) == GlobalVariableForSymbol.end()) {
    GlobalVariableForSymbol[symbol] = IRB.CreateGlobalStringPtr(symbol, "__kerma_sym");
  }

  auto PtrToStore = IRB.CreatePtrToInt(Ptr,  IRB.getInt64Ty());
  char AcccessType = op == MemOp::Load? 0 : (op == MemOp::Store? 1 : 2);

  if ( Mode == BLOCK_MODE ) {
    if ( Function *TraceFun = M->getFunction("__kerma_rec_access_b") ) {
      ArrayRef<Value*> Args({ /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AcccessType),
                              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
                              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
                              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
                              /* sym */ GlobalVariableForSymbol[symbol],
                              /* loc */ PtrToStore});
      CallInst *CI = CallInst::Create(TraceFun, Args, "", InsertBefore);
      CI->setCallingConv(CallingConv::PTX_Device);
      return true;
    }
  }

  if ( Mode == WARP_MODE ) {
    if ( Function *TraceFun = M->getFunction("__kerma_rec_access_w") ) {
      ArrayRef<Value*> Args({ /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AcccessType),
                              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
                              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
                              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
                              /* sym */ GlobalVariableForSymbol[symbol],
                              /* loc */ PtrToStore});
      CallInst *CI = CallInst::Create(TraceFun, Args, "", InsertBefore);
      CI->setCallingConv(CallingConv::PTX_Device);
      return true;
    }
  }

  if ( Mode == THREAD_MODE ) {
    if ( Function *TraceFun = M->getFunction("__kerma_rec_access_b_t") ) {
      ArrayRef<Value*> Args({ /* ty  */ ConstantInt::get(IRB.getInt8Ty(), AcccessType),
                              /* bid */ ConstantInt::get(IRB.getInt32Ty(), 0),
                              /* tid */ ConstantInt::get(IRB.getInt32Ty(), 0),
                              /* line*/ ConstantInt::get(IRB.getInt32Ty(), Loc.line),
                              /* col */ ConstantInt::get(IRB.getInt32Ty(), Loc.col),
                              /* sym */ GlobalVariableForSymbol[symbol],
                              /* loc */ PtrToStore});
      CallInst *CI = CallInst::Create(TraceFun, Args, "", InsertBefore);
      CI->setCallingConv(CallingConv::PTX_Device);
      return true;
    }
  }

  return false;
}

/// Instrument a kernel function and return
/// some statistics
PassStats InstruMemOpPrintfPass::instrumentKernel(Function &F, unsigned char KernelID) {

#ifdef KERMA_OPT_PLUGIN
  llvm::errs() << "--Instrumenting " << demangle(F.getName().str()) << "\n";
#endif

  PassStats Stats;

  Stats.changes |= instrumentKernelMeta(F, KernelID);

  for ( auto& BB : F) {
    for ( auto& I : BB) {

      SourceLoc Loc;
      if ( auto& DL = I.getDebugLoc()) {
        Loc.line = DL->getLine();
        Loc.col = DL.getCol();
      }

      if ( auto* LI = dyn_cast<LoadInst>(&I)) {
        Stats.Loads++;
        if ( instrumentAccess(F.getParent(), KernelID, LI->getPointerOperand(), Loc, LI, MemOp::Load))
          Stats.InstruLoads++;
        else
          Stats.Failed.push_back(LI);
      }
      else if ( auto* SI = dyn_cast<StoreInst>(&I)) {
        Stats.Stores++;
        if ( instrumentAccess(F.getParent(), KernelID, SI->getPointerOperand(), Loc, SI, MemOp::Store))
          Stats.InstruStores++;
        else
          Stats.Failed.push_back(SI);
      }
      else if ( auto* CI = dyn_cast<CallInst>(&I)) {
        if ( isAtomic(*CI->getCalledFunction())) {
          Stats.Atomics++;
          if ( instrumentAccess(F.getParent(), KernelID, CI->getArgOperand(0), Loc, CI, MemOp::Atomic))
            Stats.InstruAtomics++;
          else
            Stats.Failed.push_back(CI);
        }
        // TODO 1. memcpy?
        // if ( CI->getCalledFunction() && CI->getCalledFunction()->getName().startswith("malloc"))
        //   llvm::errs() << *CI << " -- " << getAddressSpace(CI).name() << "\n";
      }
    }
  }

  return Stats;
}

void InstruMemOpPrintfPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
  AU.addRequired<TypeCheckerPass>();
}

bool InstruMemOpPrintfPass::runOnModule(Module &M) {

  if ( M.getTargetTriple().find("nvptx") == std::string::npos)
    return false;

  if ( !getAnalysis<TypeCheckerPass>().moduleTypechecks())
    return false;

  CanonicalizerPass Canonicalizer;
  Canonicalizer.runOnModule(M);

  PassStats ModuleStats;

  if ( !KermaRTLinked(M)) {
#ifdef KERMA_OPT_PLUGIN
    llvm::report_fatal_error("KermaDeviceRT not found in " + M.getName());
#else
    LinkDeviceRTPass LinkKermaRTDevice;
    LinkKermaRTDevice.runOnModule(M);
#endif
  }

  InstrumentedFunctions.clear();

#ifdef KERMA_OPT_PLUGIN
  Mode = InstruMode.getValue();
  if ( !InstruTarget.getValue().empty()) {
    auto vals = parseDelimStr(InstruTarget, ',');
    for ( auto&& val : vals)
      Targets.push_back(val);
  }
#endif

  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();


#ifdef KERMA_OPT_PLUGIN
  errs() << '[' << formatv("{0,15}", "Instrumenter") << "] ";
  if ( !Kernels.size())
    errs() << "No kernels found!\n";
  else
    errs() << "Running on " << Kernels.size() << " kernels...\n";
#endif

  unsigned char id=0;
  for ( auto* kernel : Kernels) {
//     auto stats = instrumentKernel(*kernel, id++);
//     ModuleStats << stats;

//     if ( stats.InstruLoads || stats.InstruStores || stats.InstruAtomics)
//       InstrumentedFunctions.insert(kernel);

// #ifdef KERMA_OPT_PLUGIN
//   llvm::errs() << "\n>>> " << demangle(kernel->getName())
//                << " -- L:" << stats.Loads << '/' << stats.InstruLoads
//                <<   ", S:" << stats.Stores << '/' << stats.InstruStores
//                <<   ", A:" << stats.Atomics << '/' << stats.InstruAtomics << "\n\n";
// #endif
  }

  return ModuleStats.changes ||
       ( ModuleStats.InstruLoads + ModuleStats.InstruStores
                                 + ModuleStats.InstruAtomics);
}

} // namespace kerma

static RegisterPass<kerma::InstruMemOpPrintfPass> RegisterMemOpInstrumentationPass(
        /* arg      */ "kerma-mo-instru",
        /* name     */ "Instrument memory operations in CUDA kernels",
        /* CFGOnly  */ false,
        /* analysis */ false);