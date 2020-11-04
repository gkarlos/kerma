#include "kerma/Analysis/Typecheck.h"
#include "kerma/Analysis/DetectKernels.h"
#include "kerma/NVVM/NVVMUtilities.h"
#include "kerma/Utils/LLVMMetadata.h"
#include "kerma/Utils/LLVMShorthands.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Type.h>
#include <llvm/PassSupport.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/WithColor.h>

#include <string>

using namespace llvm;

namespace kerma {

static struct  {
  unsigned MaxSupportedIndices = 2;
} Opts;


// Err
enum StructErr : unsigned {
  ER_None=0,
  ER_NestedStructNotSimple,
  ER_StructFieldIsPtr,
  ER_StructFieldIsArr
};

static const std::string StructErrToStr(StructErr Err) {
  switch ( Err) {
    case ER_NestedStructNotSimple:
      return "\'Inner struct must only contain scalars\'";
    case ER_StructFieldIsArr:
    case ER_StructFieldIsPtr:
      return "\'Struct fields cannot be arrays\'";
    default:
      return "";
  }
}

TypeCheckError::TypeCheckError(const std::string& Msg, unsigned int Line, unsigned int Col)
: Msg(Msg), Line(Line), Col(Col) {}

const std::string TypeCheckError::getMsg() { return Msg; }

const std::string TypeCheckError::getMsgWithSourceLoc() {
  if ( Line && Col) {
    return Msg + " at " + std::to_string(Line) + ':' + std::to_string(Col);
  }
  else if ( Line) {
    return Msg + " at line " + std::to_string(Line);
  }
  else {
    return Msg;
  }
}


TypeCheckError TypeCheckError::getUnsupportedArrayOfPointers(Argument& Arg) {
  std::string msg = "Unsupported array of pointers for arg #" + std::to_string(Arg.getArgNo());
  if ( !Arg.getName().empty())
    msg += " (" + Arg.getName().str() + ')';
  auto *MD = findMDForArgument(&Arg);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedArrayOfPointers(GlobalVariable& GV) {
  std::string msg = "Unsupported array of pointers for global variable";
  if ( !GV.getName().empty())
    msg += " (" + GV.getName().str() + ')';
  unsigned int lineno = 0;
  auto *MD = findMDForGlobal(&GV);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedArrayOfPointers(llvm::AllocaInst &AI) {
  std::string msg = "Unsupported array of pointers for local variable";
  if ( !AI.getName().empty())
    msg += " (" + AI.getName().str() + ')';
  auto *MD = findMDForAlloca(&AI);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedDimensions(GlobalVariable& GV, unsigned int Dims) {
  std::string msg = "Unsupported number of dimensions(" + std::to_string(Dims) + ") for global";
  if ( !GV.getName().empty())
    msg += " (" + GV.getName().str() + ')';
  unsigned int lineno = 0;
  auto *MD = findMDForGlobal(&GV);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedDimensions(Argument& Arg, unsigned int Dims) {
  std::string msg = "Unsupported number of dimensions(" + std::to_string(Dims) + ") for arg #" + std::to_string(Arg.getArgNo());
  if ( !Arg.getName().empty())
    msg += " (" + Arg.getName().str() + ')';
  auto *MD = findMDForArgument(&Arg);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedDimensions(AllocaInst &AI, unsigned int Dims) {
  std::string msg = "Unsupported number of dimensions(" + std::to_string(Dims) + ") for local variable";
  if ( !AI.getName().empty())
    msg += " (" + AI.getName().str() + ')';
  auto *MD = findMDForAlloca(&AI);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedDimensions(CallInst& CI, unsigned int Dims) {
  std::string msg = "Unsupported number of dimensions(" + std::to_string(Dims) + ") for malloc result";
  auto& DL = CI.getDebugLoc();
  return TypeCheckError(msg, DL? DL->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(GlobalVariable &GV, StructErr Err) {
  assert(Err);

  std::string msg = "Unsupported struct type in global ";
  if ( !GV.getName().empty())
    msg += "(" + GV.getName().str() + ") ";
  msg += StructErrToStr(Err);
  auto *MD = findMDForGlobal(&GV);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(Argument &Arg, StructErr Err) {
  assert(Err);
  std::string msg = "Unsupported struct type in arg #" + std::to_string(Arg.getArgNo());
  if ( !Arg.getName().empty())
    msg += " (" + Arg.getName().str() + ") ";
  msg += StructErrToStr(Err);
  auto *MD = findMDForArgument(&Arg);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(AllocaInst& AI, StructErr Err) {
  std::string msg = "Unsupported struct type in local variable";
  if ( !AI.getName().empty())
    msg += " (" + AI.getName().str() + ") ";
  msg += StructErrToStr(Err);
  auto *MD = findMDForAlloca(&AI);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(CallInst& CI, StructErr Err) {
  std::string msg = "Unsupported struct type for malloc result";
  auto& DL = CI.getDebugLoc();
  return TypeCheckError(msg, DL? DL->getLine() : 0);
}

// Util
static StructErr typecheckStruct(StructType *Ty) {

  for ( unsigned int i = 0; i < Ty->getNumElements(); ++i) {
    auto *FieldTy = Ty->getElementType(i);

    // pointer/array in struct
    if ( FieldTy->isPointerTy())
      return ER_StructFieldIsPtr;
    if ( FieldTy->isArrayTy())
      return ER_StructFieldIsArr;

    // struct in struct
    if ( FieldTy->isStructTy() && !isSimpleStruct(dyn_cast<StructType>(FieldTy)))
      return ER_NestedStructNotSimple;
  }

  return ER_None;
}

static bool typecheckKernelArgs(llvm::Function& Kernel, SmallVectorImpl<TypeCheckError>& Errors) {
  auto pre = Errors.size();

  for ( auto& Arg : Kernel.args()) {

    auto *ArgTy = Arg.getType();

    /// Arg is star star, e.g: int **X;
    if ( auto *PtrTy = dyn_cast<PointerType>(ArgTy)) {
      if ( PtrTy->getElementType()->isPointerTy() || PtrTy->getElementType()->isArrayTy())
        Errors.push_back(TypeCheckError::getUnsupportedArrayOfPointers(Arg));
    }
    /// Arg is an array literal
    else if ( auto ArrayTy = dyn_cast<ArrayType>(ArgTy)) {
      /// Array of pointers, e.g: int *X[15]
      if ( stripArrayNest(ArrayTy)->isPointerTy())
        Errors.push_back(TypeCheckError::getUnsupportedArrayOfPointers(Arg));
      /// 3D Array, e.g: int X[15][15][15]
      else if (auto MaxIndices = getMaxIndicesForType(ArgTy);
                    MaxIndices > Opts.MaxSupportedIndices)
        Errors.push_back(TypeCheckError::getUnsupportedDimensions(Arg, MaxIndices));
    }

    if ( ArgTy->isPointerTy() || ArgTy->isArrayTy()) {
      /// single pointer or max 2D array. Now check the element type
      if (auto* ElemTy = isa<PointerType>(ArgTy)? stripPointers(ArgTy) : stripArrayNest(ArgTy);
                ElemTy->isStructTy()) {
        if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
          Errors.push_back(TypeCheckError::getUnsupportedStruct(Arg, Err));
      }
    }

    else if ( ArgTy->isStructTy()) {
      /// This is probably not needed as Clang lowers struct args to
      /// struct pointers with Attribute::ByVal so it 'should' be
      /// handled by the previous check.
      /// But lets just check for it anyway
      if ( auto Err = typecheckStruct(dyn_cast<StructType>(ArgTy)))
        Errors.push_back(TypeCheckError::getUnsupportedStruct(Arg, Err));
    }
  }

  return Errors.size() == pre;
}

static bool typecheckKernelLocals( Function& Kernel, SmallVectorImpl<TypeCheckError>& Errors) {
  auto pre = Errors.size();

  for ( auto& BB : Kernel) {
    for ( auto& I : BB) {
      if ( auto *AI = dyn_cast<AllocaInst>(&I)) {

        auto *AllocatedTy = AI->getAllocatedType();

        /// Array Alloca
        if ( AllocatedTy->isArrayTy()) {

          auto *ElemTy = stripArrayNest(AllocatedTy);

          /// Array of ptrs
          if ( ElemTy->isPointerTy())
            Errors.push_back(TypeCheckError::getUnsupportedArrayOfPointers(*AI));

          /// 3D Array
          else if (auto MaxIndices = getMaxIndicesForType(AllocatedTy);
                        MaxIndices > Opts.MaxSupportedIndices)
            Errors.push_back(TypeCheckError::getUnsupportedDimensions(*AI, MaxIndices));

          /// Array of structs
          else if ( ElemTy->isStructTy())
            if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
              Errors.push_back(TypeCheckError::getUnsupportedStruct(*AI, Err));
        }

        /// Struct Alloca
        else if ( AllocatedTy->isStructTy()) { // local struct
          if ( auto Err = typecheckStruct(dyn_cast<StructType>(AllocatedTy)))
            Errors.push_back(TypeCheckError::getUnsupportedStruct(*AI, Err));
        }
      }
      else if ( auto *CI = dyn_cast<CallInst>(&I)) {
        auto *Callee = CI->getCalledFunction();
        if ( Callee->getName().equals("malloc")) { // malloc result
          for ( auto *U : CI->users()) {
            if ( auto *BitCast = dyn_cast<BitCastInst>(U)) {
              if ( auto MaxIndices = getMaxIndicesForType(BitCast->getDestTy()); MaxIndices > Opts.MaxSupportedIndices)
                Errors.push_back(TypeCheckError::getUnsupportedDimensions(*CI, MaxIndices));
              else if ( auto *ElemTy = stripArrayNest(BitCast->getDestTy()); ElemTy->isStructTy())
                if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
                  Errors.push_back(TypeCheckError::getUnsupportedStruct(*CI, Err));
            }
          }
        }
      }
    }
  }
  return Errors.size() == pre;
}

// API
bool isPtrToStruct(Type *Ty) {
  if ( auto *Ptr = dyn_cast<PointerType>(Ty))
    return Ptr->getElementType()->isStructTy();
  return false;
}

bool isStructOfScalars(StructType *Ty) {
  for ( auto *ElemTy : Ty->elements())
    if ( ElemTy->isAggregateType() || ElemTy->isVectorTy() || ElemTy->isPointerTy())
      return false;
  return true;
}

bool isSimpleStruct(StructType *Ty) {
  for ( unsigned int i = 0; i < Ty->getNumElements(); ++i) {
    auto *FieldTy = Ty->getElementType(i);
    if ( FieldTy->isPointerTy() || FieldTy->isAggregateType())
      return false;
  }
  return true;
}

// Type* stripPointers(Type *Ty) {
//   Type *tmp = Ty;
//   while ( auto *ptr = dyn_cast<PointerType>(tmp))
//     tmp = ptr->getElementType();
//   return tmp;
// }

Type *stripArrayNest(Type *Ty) {
  Type *tmp = Ty;
  while ( auto *ptr = dyn_cast<ArrayType>(tmp))
    tmp = ptr->getElementType();
  return tmp;
}

unsigned int getMaxIndicesForType( Type *Ty) {
  // if ( !Ty && !isa<PointerType>(Ty) && !isa<ArrayType>(Ty))
  //   return 0;

  Type *tmp = Ty;
  unsigned int MaxIndices = 0;

  while ( true) {
    // array of pointers
    if ( auto *ElemPtrTy = dyn_cast<PointerType>(tmp)) {
      MaxIndices++;
      tmp = ElemPtrTy->getElementType();
    }
    else if ( auto *ElemArrTy = dyn_cast<ArrayType>(tmp)) {
      MaxIndices++;
      tmp = ElemArrTy->getElementType();
    }
    else {
      break;
    }

  }
  return MaxIndices;
}

bool typecheckGlobals(Module& M, llvm::SmallVectorImpl<TypeCheckError>& Errors) {
  auto pre = Errors.size();

  for ( auto& Global : M.globals()) {
    if ( auto* GV = dyn_cast<GlobalVariable>(&Global)) {

      if ( nvvm::isNVVMSymbol(GV->getName()))
        continue;

      // All globals are pointers, get the element type
      auto *GVTy = GV->getType()->getElementType();

      /// star star, e.g: int **X;
      if ( GVTy->isPointerTy() && dyn_cast<PointerType>(GVTy)->getElementType()->isPointerTy())
        Errors.push_back(TypeCheckError::getUnsupportedArrayOfPointers(*GV));
      else if ( GVTy->isArrayTy()) {
        /// array of ptrs, e.g: int *X[42];
        if ( stripArrayNest(GVTy)->isPointerTy())
          Errors.push_back(TypeCheckError::getUnsupportedArrayOfPointers(*GV));
        else if (auto MaxIndices = getMaxIndicesForType(GVTy);
                      MaxIndices > Opts.MaxSupportedIndices)
          Errors.push_back(TypeCheckError::getUnsupportedDimensions(*GV, MaxIndices));
      }
      else if ( GVTy->isPointerTy() || GVTy->isArrayTy()) {
        // pointer/array is max 2D. Now check the element type
        if (auto *ElemTy = isa<PointerType>(GVTy)? stripPointers(GVTy) : stripArrayNest(GVTy);
                 ElemTy->isStructTy()) {
          if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
            Errors.push_back(TypeCheckError::getUnsupportedStruct(*GV, Err));
        }
      }
      else if ( GVTy->isStructTy())
        if ( auto Err = typecheckStruct(dyn_cast<StructType>(GVTy)))
          Errors.push_back(TypeCheckError::getUnsupportedStruct(*GV, Err));
    }
  }
  return Errors.size() == pre;
}

bool typecheckKernel(llvm::Function& F, llvm::SmallVectorImpl<TypeCheckError>& Errors) {
  /// Avoid short-circuiting to report more errors
  auto args = typecheckKernelArgs(F, Errors);
  auto lcls = typecheckKernelLocals(F, Errors);
  return args && lcls;
}

// Pass
char TypeCheckerPass::ID = 11;

TypeCheckerPass::TypeCheckerPass() : ModulePass(ID) {}

SmallVector<TypeCheckError, 32> TypeCheckerPass::getErrors() { return Errors; }

bool TypeCheckerPass::moduleTypechecks() { return Errors.empty(); }

void TypeCheckerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
}

void TypeCheckerPass::dumpErrors() {
  WithColor(errs(), HighlightColor::Note) << '[';
  WithColor(errs(), raw_ostream::Colors::GREEN) << formatv("{0,15}", "Typechecker");
  WithColor(errs(), HighlightColor::Note) << ']';

  if ( Errors.empty())
   errs() << " Yes" << "\n";
  else {
    WithColor(errs(), raw_ostream::Colors::RED, true) << " No" << "\n";
    for ( auto& Err : Errors) {
      WithColor::warning() << Err.getMsgWithSourceLoc() << "\n";
    }
  }
}

bool
TypeCheckerPass::runOnModule(llvm::Module &M) {
  Errors.clear();
  typecheckGlobals(M, Errors);
  for ( auto Kernel : getAnalysis<DetectKernelsPass>().getKernels() )
    typecheckKernel(*Kernel.getFunction(), Errors);

#ifdef KERMA_OPT_PLUGIN
  dumpErrors();
#endif
  return false;
}

namespace {

static RegisterPass<TypeCheckerPass> RegisterTypeChekerPass(
        /* arg      */ "kerma-tc",
        /* name     */ "Check for unsupported memory types",
        /* CFGOnly  */ false,
        /* analysis */ true);

} // anonymous namespace
} // namespace kerma