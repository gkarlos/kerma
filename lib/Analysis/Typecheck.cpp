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
#include <llvm/Support/raw_ostream.h>

#include <string>

using namespace llvm;

namespace kerma {


bool isPtrToStruct(Type *Ty) {
  if ( auto *Ptr = dyn_cast<PointerType>(Ty))
    return Ptr->getElementType()->isStructTy();
  return false;
}

bool isStructOfScalars(StructType *Ty) {
  for ( auto *ElemTy : Ty->elements())
    if ( ElemTy->isAggregateType() || ElemTy->isPointerTy())
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

Type* stripPointers(Type *Ty) {
  Type *tmp = Ty;
  while ( auto *ptr = dyn_cast<PointerType>(tmp))
    tmp = ptr->getElementType();
  return tmp;
}

Type *stripArrayNest(Type *Ty) {
  Type *tmp = Ty;
  while ( auto *ptr = dyn_cast<ArrayType>(tmp))
    tmp = ptr->getElementType();
  return tmp;
}

unsigned int getMaxIndicesForType( Type *Ty) {
  if ( !Ty && !isa<PointerType>(Ty) && !isa<ArrayType>(Ty))
    return 0;

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


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

TypeCheckError::TypeCheckError(const std::string& Msg, unsigned int Line, unsigned int Col)
: Msg(Msg), Line(Line), Col(Col) {}

const std::string
TypeCheckError::getMsg() { return Msg; }

const std::string
TypeCheckError::getMsgWithSourceLoc() {
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

static const std::string StructErrToStr(StructErr Err) {
  switch ( Err) {
    case ER_NestedStructNotSimple:
      return " \'Inner struct must only contain scalars\'";
    case ER_StructFieldHasUnsupportedDims:
      return " \'Struct field has unsupported dimensions (>" + std::to_string(TypeCheck::MaxSupportedIndices) + ")\'";
    case ER_StructFieldIsPtrToUnsupportedStruct:
      return " \'Struct field is pointer to unsupported struct\'";
    case ER_StructFieldIsArrOfUnsupportedStructs:
      return " \'Struct field is array of unsupported structs\'";
    default:
      return "";
  }
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

  std::string msg = "Unsupported struct type in global";
  if ( !GV.getName().empty())
    msg += " (" + GV.getName().str() + ')';
  msg += StructErrToStr(Err);
  auto *MD = findMDForGlobal(&GV);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(llvm::Argument &Arg, StructErr Err) {
  assert(Err);

  std::string msg = "Unsupported struct type in arg #" + std::to_string(Arg.getArgNo());
  if ( !Arg.getName().empty())
    msg += " (" + Arg.getName().str() + ')';

  msg += StructErrToStr(Err);

  auto *MD = findMDForArgument(&Arg);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(AllocaInst& AI, StructErr Err) {
  std::string msg = "Unsupported struct type in local variable";
  if ( !AI.getName().empty())
    msg += " (" + AI.getName().str() + ')';
  msg += StructErrToStr(Err);
  auto *MD = findMDForAlloca(&AI);
  return TypeCheckError(msg, MD? MD->getLine() : 0);
}

TypeCheckError TypeCheckError::getUnsupportedStruct(CallInst& CI, StructErr Err) {
  std::string msg = "Unsupported struct type for malloc result";
  auto& DL = CI.getDebugLoc();
  return TypeCheckError(msg, DL? DL->getLine() : 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

char TypeCheckerPass::ID = 11;

TypeCheckerPass::TypeCheckerPass() : ModulePass(ID) {}

void
TypeCheckerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetectKernelsPass>();
}


SmallVector<TypeCheckError, 32> TypeCheckerPass::getErrors() { return Errors; }

void TypeCheckerPass::dumpErrors() {
  if ( Errors.empty())
    llvm::errs() << "\n[Typechecker] Pass!" << "\n";
  else {
    llvm::errs() << "\n[Typechecker] Found " << Errors.size() << " errors\n";
    for ( auto& Err : Errors) {
      llvm::errs() << " - " << Err.getMsgWithSourceLoc() << "\n";
    }
  }
}

bool TypeCheckerPass::moduleTypechecks() { return Errors.empty(); }

void TypeCheckerPass::typecheckGlobals(Module& M) {
  for ( auto& Global : M.globals()) {
    if ( auto* GV = dyn_cast<GlobalVariable>(&Global)) {
      if ( !nvvm::isNVVMSymbol(GV->getName())) {
        auto *GVTy = GV->getType()->getElementType();

        if ( GVTy->isPointerTy() || GVTy->isArrayTy()) {
          if ( auto MaxIndices = getMaxIndicesForType(GVTy); MaxIndices > TypeCheck::MaxSupportedIndices)
            Errors.push_back(TypeCheckError::getUnsupportedDimensions(*GV, MaxIndices));

          // pointer/array is max 2D. Now check the element type
          else if (auto *ElemTy = isa<PointerType>(GVTy)? stripPointers(GVTy) : stripArrayNest(GVTy);
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
  }
}

StructErr TypeCheckerPass::typecheckStruct(StructType *Ty) {

  for ( unsigned int i = 0; i < Ty->getNumElements(); ++i) {
    auto *FieldTy = Ty->getElementType(i);

    // pointer/array in struct
    if ( FieldTy->isPointerTy() || FieldTy->isArrayTy()) {
      if ( getMaxIndicesForType(FieldTy) > TypeCheck::MaxSupportedIndices)
        return ER_StructFieldHasUnsupportedDims;

      auto *ElemTy = isa<PointerType>(FieldTy)? stripPointers(FieldTy)
                                              : stripArrayNest(FieldTy);

      // pointer/array to struct -> Must only be simple
      if ( ElemTy->isStructTy() && !isSimpleStruct(dyn_cast<StructType>(ElemTy)))
        return FieldTy->isPointerTy()? ER_StructFieldIsPtrToUnsupportedStruct
                                     : ER_StructFieldIsArrOfUnsupportedStructs;
    }

    // struct in struct
    if ( FieldTy->isStructTy() && !isSimpleStruct(dyn_cast<StructType>(FieldTy)))
      return ER_NestedStructNotSimple;
  }

  return ER_None;
}

void TypeCheckerPass::typecheckKernelArgs(llvm::Function& Kernel) {

  for ( auto& Arg : Kernel.args()) {

    auto *ArgTy = Arg.getType();

    // pointer/array argument
    if ( ArgTy->isPointerTy() || ArgTy->isArrayTy()) {
      if ( auto MaxIndices = getMaxIndicesForType(ArgTy); MaxIndices > TypeCheck::MaxSupportedIndices)
        Errors.push_back(TypeCheckError::getUnsupportedDimensions(Arg, MaxIndices));

      // pointer/array is max 2D. Now check the element type
      else if (auto *ElemTy = isa<PointerType>(ArgTy)? stripPointers(ArgTy) : stripArrayNest(ArgTy);
          ElemTy->isStructTy()) {
        if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
          Errors.push_back(TypeCheckError::getUnsupportedStruct(Arg, Err));
      }
    }

    else if ( ArgTy->isStructTy()) {
      // This is probably not needed as Clang lowers struct args to
      // struct pointers with Attribute::ByVal so it 'should' be
      // handled by the previous check.
      // But lets just check for it anyway
      if ( auto Err = typecheckStruct(dyn_cast<StructType>(ArgTy)))
        Errors.push_back(TypeCheckError::getUnsupportedStruct(Arg, Err));
    }
  }
}

void TypeCheckerPass::typecheckKernelLocals(llvm::Function& Kernel) {
  for ( auto& BB : Kernel) {
    for ( auto& I : BB) {
      if ( auto *AI = dyn_cast<AllocaInst>(&I)) {

        auto *AllocatedTy = AI->getAllocatedType();

        if ( AllocatedTy->isArrayTy()) { // local array
          if ( auto MaxIndices = getMaxIndicesForType(AllocatedTy); MaxIndices > TypeCheck::MaxSupportedIndices)
            Errors.push_back(TypeCheckError::getUnsupportedDimensions(*AI, MaxIndices));
          else if ( auto *ElemTy = stripArrayNest(AI->getAllocatedType()); ElemTy->isStructTy())
            if ( auto Err = typecheckStruct(dyn_cast<StructType>(ElemTy)))
              Errors.push_back(TypeCheckError::getUnsupportedStruct(*AI, Err));
        }
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
              if ( auto MaxIndices = getMaxIndicesForType(BitCast->getDestTy()); MaxIndices > TypeCheck::MaxSupportedIndices)
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
}

void TypeCheckerPass::typecheckKernels(Module& M) {
  auto Kernels = getAnalysis<DetectKernelsPass>().getKernels();

  for ( auto *Kernel : Kernels) {
    typecheckKernelArgs(*Kernel);
    typecheckKernelLocals(*Kernel);
  }
}

bool
TypeCheckerPass::runOnModule(llvm::Module &M) {
  typecheckKernels(M);
  typecheckGlobals(M);
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