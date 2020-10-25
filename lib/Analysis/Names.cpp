#include "kerma/Analysis/Names.h"
#include "kerma/Utils/LLVMMetadata.h"

// #include "kerma/Analysis/DetectKernels.h"
// #include "kerma/Analysis/Typecheck.h"
// #include "kerma/NVVM/NVVMUtilities.h"
// #include "kerma/Support/CXXExtras.h"
// #include "kerma/Support/Demangle.h"
// #include "kerma/Transforms/Canonicalize/GepifyMem.h"
// #include "kerma/Transforms/Canonicalize/SimplifyGEP.h"
// #include "kerma/Utils/LLVMShorthands.h"
// #include "kerma/Analysis/DataDependency.h"

// #include <llvm/Analysis/LoopInfo.h>
// #include <llvm/Analysis/ScalarEvolution.h>
// #include <llvm/Analysis/ValueTracking.h>
// #include <llvm/ADT/SmallSet.h>
// #include <llvm/ADT/SmallVector.h>
// #include <llvm/ADT/StringRef.h>
// #include <llvm/ADT/SmallString.h>
// #include <llvm/IR/Argument.h>
// #include <llvm/IR/Attributes.h>
// #include <llvm/IR/Constant.h>
// #include <llvm/IR/Constants.h>
// #include <llvm/IR/DataLayout.h>
// #include <llvm/IR/DebugInfoMetadata.h>
// #include <llvm/IR/DerivedTypes.h>
// #include <llvm/IR/Function.h>
// #include <llvm/IR/GlobalVariable.h>
// #include <llvm/IR/InstrTypes.h>
// #include <llvm/IR/Instruction.h>
// #include <llvm/IR/Instructions.h>
// #include "llvm/IR/IntrinsicsNVPTX.h"
// #include <llvm/IR/IntrinsicInst.h>
// #include <llvm/IR/Intrinsics.h>
// #include <llvm/IR/Metadata.h>
// #include <llvm/Support/Casting.h>
// #include <llvm/Support/ErrorHandling.h>
// #include <llvm/Support/FormatVariadic.h>
// #include <llvm/Support/raw_ostream.h>
// #include <llvm/Analysis/MemorySSA.h>
// #include <llvm/PassSupport.h>
#include <string>
#include <type_traits>
#include <utility>

using namespace kerma;
using namespace llvm;

static std::map<Value *, std::string> NameCache;

static int AllocaCount = 0;
static int ArgCount = 0;
static int GlbCount = 0;
static int TmpCount = 0;
static int UnknownCount = 0;


/// Linear search of the debug info of a function F for an entry
/// corresponding to the value V. If found, the name is written
/// to Res. if non empty true is returned. False otherwise
// static bool searchDebugInfo(Function *F, Value *V, std::string& Res) {
//   if ( !F)
//     return false;

//   for ( auto& BB : *F)
//     for ( auto& I : BB) {
//       if ( auto *DbgDecl = dyn_cast<DbgDeclareInst>(&I);
//             DbgDecl && (DbgDecl->getAddress() == V))
//       {
//         if ( auto name = DbgDecl->getVariable()->getName(); !name.empty())
//           Res.assign(name);
//         goto EXIT_SEARCH;
//       }
//       if ( auto *DbgVal = dyn_cast<DbgValueInst>(&I);
//             DbgVal && (DbgVal->getValue() == V))
//       {
//         if ( auto name = DbgVal->getVariable()->getName(); !name.empty())
//           Res.assign(name);
//         goto EXIT_SEARCH;
//       }
//     }

// EXIT_SEARCH:
//   return !Res.empty();
// }

/// Search for a name in debug info. If found retrieve it, otherwise if not
/// found and Generate == true, generate a unique name for the Alloca.
///
/// This function should be called only when AI.getName().empty() is true,
/// i.e it will not perform that check
// static bool getNameForAlloca(AllocaInst *AI, bool Generate, std::string& Res) {
//   assert(AI && "AllocaInst cannot be null");

//   if ( auto *BB = AI->getParent())
//     if ( auto *F = BB->getParent())
//       searchDebugInfo(F, AI, Res);

//   // we found something in debug info
//   if ( !Res.empty())
//     return true;

//   if ( Generate) {
//     Res += "loc" + std::to_string(AllocaCount++);
//     NameCache.emplace(AI, Res);
//   }
//   return false;
// }

/// Get a name for a global variable
///
/// This function should be called only when getName().empty() is true,
/// i.e it will not perform that check
// static bool getNameForGlobal(GlobalVariable *GV, bool Generate, std::string& Res) {
//   assert(GV && "GlobalVariable cannot be null");

//   SmallVector<DIGlobalVariableExpression*, 4> GVExprs;
//   GV->getDebugInfo(GVExprs);

//   auto* MDNode = GV->getMetadata("dbg");

//   if ( auto* DIExpr = dyn_cast<DIGlobalVariableExpression>(MDNode)) {
//     auto name = DIExpr->getVariable()->getDisplayName();
//     if ( !name.empty()) {
//       Res += name;
//       return true;
//     }
//   }

//   if ( Generate)
//     Res += "glb" + std::to_string(GlbCount++);

//   return false;
// }

/// Get the name of a function Argument
// static bool getNameForArg(Argument *Arg, bool Generate, std::string& Res) {
//   assert(Arg && "Argument cannot be null");

//   if ( auto *F = Arg->getParent())
//     searchDebugInfo(F, Arg, Res);

//   if ( !Res.empty())
//     return true;

//   if ( Generate) {
//     Res += "arg" + std::to_string(ArgCount++) + "." + std::to_string(Arg->getArgNo());
//     NameCache.emplace(Arg, Res);
//   }
//   return false;
// }

/// Return true if a name was found. False otherwise
/// If a name is generated false is returned
// static bool getNameForValue(Value *V, bool Generate, std::string& Res) {
//   if ( auto entry = NameCache.find(V); entry != NameCache.end()) {
//     Res += entry->second;
//     return true;
//   }

//   if ( auto *Arg = dyn_cast<Argument>(V))
//     return getNameForArg(Arg, Generate, Res);
//   else if ( auto *GV = dyn_cast<GlobalVariable>(V))
//     return getNameForGlobal(GV, Generate, Res);

//   else if ( auto *I = dyn_cast<Instruction>(V)) {

//     if ( auto *AI = dyn_cast<AllocaInst>(I))
//       return getNameForAlloca(AI, Generate, Res);

//     if ( auto *LI = dyn_cast<LoadInst>(I))
//       return getNameForValue(LI->getPointerOperand(), Generate, Res);
//     if ( auto *SI = dyn_cast<StoreInst>(I))
//       return getNameForValue(SI->getPointerOperand(), Generate, Res);
//     if ( auto *CI = dyn_cast<CallInst>(I)) {
//       if ( nvvm::isNVVMIntrinsic(*CI->getCalledFunction()))
//         return getNameForNVVMIntrinsic(CI->getCalledFunction(), Res);
//       if ( nvvm::isNVVMAtomic(*CI->getCalledFunction()))
//         return getNameForValue(CI->getArgOperand(0), Generate, Res);
//     }

//     if ( auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
//       auto PtrOperand = GEP->getPointerOperand();

//       if ( auto PtrTy = dyn_cast<PointerType>(PtrOperand->getType())) {

//         // pointer to pointer
//         if ( PtrTy->getElementType()->isPointerTy()) {

//         }
//         else {
//         }



//         // if ( OperandTy)

//       if ( auto PtrTy = dyn_cast<PointerType>(PtrOperand->getType()))
//         if ( PtrTy->getElementType()->isStructTy())
//           Res += "->";
//       }
//     }


//   }

//   return false;
// }

// std::string getNameForValue(Value *V, bool Generate) {
//   if ( !V)
//     return Generate? ("unknw" + std::to_string(UnknownCount++)) : "";
//   std::string Res;
//   getNameForValue(V, Generate, Res);
//   return Res;
// }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// char SymbolizerPass::ID = 10;

// SymbolizerPass::SymbolizerPass() : ModulePass(ID) {}


// const SymbolMap& SymbolizerPass::getSymbols() { return Symbols; }

/**
 * @brief Try retrieve the field name of a struct by index
 *
 * @param Ty The type of the struct
 * @param Index Index of the field
 * @param MDNode MDNode for a value of that type
 * @return std::string
 */
// static std::string getSymbolForStructElement(Type *Ty, bool byVal, Value *Index, MDNode *MDNode) {
//   if ( auto *StructTy = dyn_cast<StructType>(Ty)) {
//     if ( auto *ConstInt = dyn_cast<ConstantInt>(Index);
//          ConstInt && StructTy->getNumElements() >= ConstInt->getZExtValue()) {

//       DIType *DITy = nullptr;

//       if ( auto *DILV = dyn_cast<DILocalVariable>(MDNode))
//         DITy = DILV->getType();
//       else if ( auto *DIGVExpr = dyn_cast<DIGlobalVariableExpression>(MDNode))
//         DITy = DIGVExpr->getVariable()->getType();

//       if ( DITy) {
//         DICompositeType *DICompositeTy = nullptr;
//         if ( byVal)
//           DICompositeTy = dyn_cast<DICompositeType>(DITy);
//         else
//           DICompositeTy = dyn_cast<DICompositeType>(dyn_cast<DIDerivedType>(DITy)->getBaseType());

//         if ( DICompositeTy) {
//           auto Elements = DICompositeTy->getElements();
//           unsigned int Idx = 0, FieldIdx = ConstInt->getZExtValue();
//           for ( auto Elem : Elements) {
//             if ( Idx == FieldIdx)
//               if ( auto *DIDerivedTy = dyn_cast<DIDerivedType>(Elem))
//                 return DIDerivedTy->getName();
//             if ( ++Idx > FieldIdx)
//               break;
//           }
//         }
//       }
//       return std::to_string(ConstInt->getZExtValue());
//     }
//     return "tmp";
//   }
//   return "";
// }

// static bool structElementIsPtr(StructType *Ty, int idx) {
//   return Ty->getElementType(idx)->isPointerTy();
// }

/// Linear search of the debug info of a function F for an entry
/// corresponding to the value V. If found, the name is returned
/// otherwise the empty string
// static DILocalVariable* findLVMetadata(Function &F, Value &V) {
//   for ( auto& BB : F)
//     for ( auto& I : BB) {
//       if ( auto *DbgDecl = dyn_cast<DbgDeclareInst>(&I);
//             DbgDecl && (DbgDecl->getAddress() == &V))
//         return DbgDecl->getVariable();
//       if ( auto *DbgVal = dyn_cast<DbgValueInst>(&I);
//             DbgVal && (DbgVal->getValue() == &V))
//         return DbgVal->getVariable();
//     }
//   return nullptr;
// }

// static void throwIfUnsupportedStruct(Type *Ty) {
//   if ( StructType *StructTy = dyn_cast<StructType>(stripPointers(Ty))) {
//     /// Check the struct members for an unsupoported struct
//     /// or chain of pointers to unsupported struct
//     for ( auto* ElemTy : StructTy->elements()) {
//       if ( auto *StructElemTy = dyn_cast<StructType>(stripPointers(ElemTy)))
//         if ( !isStructOfScalars(StructElemTy))
//           llvm_unreachable(("UNSUPPORTED TYPE: " + StructTy->getStructName().str()).c_str());
//           // for now crash and burn. TODO: Make a pass that can recognize nested structs
//           // in globals and args before any analysis so that we can exit gracefully. e.g
//           // with a message to the user
//     }
//   }
// }

// static void propagateValueSymbol(Value *V) {
//   for ( auto *User : V->users()) {
//     User->setName(V->getName());
//   }
// }

// static void SymbolizeArg(Argument *Arg) {
//   bool isByVal = Arg->hasAttribute(Attribute::ByVal);

//   for ( auto *User : Arg->users()) {

//     std::string UserName = Arg->getName();

//     if ( auto *GEP = dyn_cast<GetElementPtrInst>(User)) {

//       auto *PtrTy = dyn_cast<PointerType>(GEP->getPointerOperandType());

//       // GEP with ptr to struct
//       if ( isPtrToStruct(PtrTy)) {
//         assert(GEP->getNumIndices() >= 2 && "GEP for struct has < 2 indices!");
//         assert(isa<ConstantInt>(GEP->getOperand(2)) && "GEP index for struct element not a ConstantInt");



//         auto *StructTy = dyn_cast<StructType>(PtrTy->getElementType());
//         auto StructElem = dyn_cast<ConstantInt>(GEP->getOperand(2))->getZExtValue();

//         std::string suffix = isByVal ? "*" : "[]";

//         // Since we catch nested structs early on, the struct member
//         // accessed is either a scalar, a pointer to a scalar, or an array of scalars;
//         // if ( structElementIsPtr(StructTy, StructElem)) {
//         GEP->setName(Arg->getName() + suffix
//                                     + "." + getSymbolForStructElement( StructTy, isByVal,
//                                                                        GEP->getOperand(2),
//                                                                        findLVMetadata(*Arg->getParent(), *Arg)) + "#");
//         // } else {
//         //   // Q: should we care if we access
//         // }
//         // for now:

//       }
//       propagateValueSymbol(User);
//     }
//   }
// }

// static bool isStructPtrArgByVal(Value *V) {
//   if ( auto *Arg = dyn_cast<Argument>(V))
//     return Arg->hasAttribute(Attribute::ByVal);
//   return false;
// }

// bool SymbolizerPass::lookup(Value *V, Function *F, std::string& Symbol) {
//   if ( auto entry = GlobalSymbols.find(V); entry != GlobalSymbols.end()) {
//     Symbol = entry->second;
//     return true;
//   }
//   if ( auto entry = LocalSymbols[F].find(V); entry != LocalSymbols[F].end()) {
//     Symbol = entry->second;
//     return true;
//   }
//   if ( auto entry = ArgSymbols[F].find(V); entry != ArgSymbols[F].end()) {
//     Symbol = entry->second;
//     return true;
//   }
//   if ( auto entry = MemAccessSymbols[F].find(V); entry != MemAccessSymbols[F].end()) {
//     Symbol = entry->second;
//     return true;
//   }
//   return false;
// }

// #define MAP_FIND(M, V) (M).find(V) != M.end()

// static unsigned getMemoryAccessSize(Instruction &I) {
//     unsigned value = 0;
//     unsigned Opcode = I.getOpcode();

//     Value *addr = (Opcode == Instruction::Load) ?
//         cast<LoadInst>(&I)->getPointerOperand() :
//         cast<StoreInst>(&I)->getPointerOperand();

//     Type *type = addr->getType();
//     if (type->isPointerTy())
//         type = type->getContainedType(0);

//     DataLayout DL = I.getParent()->getParent()->getParent()->getDataLayout();

//     value = DL.getTypeAllocSize(type);

//     if (!value)
//         value = 4; // default

//     return value;
// }

// static std::string getSymbol(Value *Ptr, const DataLayout& DL) {
//   assert(Ptr);

//   auto *Object = GetUnderlyingObject(Ptr, DL);

//   if ( Argument *Arg = dyn_cast<Argument>(Object)) {
//     if ( auto *PtrTy = dyn_cast<PointerType>(Arg->getType())) {

//       if ( getPointerDepth(*PtrTy) == 1) {
//         llvm::errs() << "PtrDepth 1\n";
//       }
//       // ptr argument
//       // if ( Arg->hasAttribute(Attribute::ByVal))
//       //   return Arg->getName();
//     }
//     if ( auto *StructTy = dyn_cast<StructType>(Arg->getType())) {
//       llvm::errs() << "Arg, Struct Literal: TODO\n";
//       return "StructLiteralArg";
//     }
//   }
//   return "";
// }

static void testValueTracking(Function& F) {
  // llvm::errs() << "\nValue tracking for: " << demangleFnWithoutArgs(F) << "\n";
  // for ( auto& BB : F) {
  //   for ( auto& I : BB) {
  //     if ( auto *LI = dyn_cast<LoadInst>(&I)) {
  //       assert(isa<GetElementPtrInst>(LI->getPointerOperand()) && "Non-GEP ptr! Run CanonLoadsAndStores pass!\n");

  //       llvm::errs() << ">>> Load Object: " << I << "\n";
  //       llvm::errs() << getSymbol(LI->getPointerOperand(), F.getParent()->getDataLayout());

  //       llvm::errs() << mayDependOnData(LI);
        // auto *Object = llvm::GetUnderlyingObject(LI->getPointerOperand(), F.getParent()->getDataLayout());
        // // llvm::errs() << "  " << *llvm::GetUnderlyingObject(LI->getPointerOperand(), F.getParent()->getDataLayout()) << "\n";
        // llvm::errs() << *Object << "\n";
        // if ( Object)
        //   llvm::errs() << getSymbol(Object, LI->getPointerOperand());
        // llvm::errs() << Object->g
      }
      // else if ( auto *SI = dyn_cast<StoreInst>(&I)) {
      //   llvm::errs() << ">>> Store Object: " << I << "\n";
      //   // llvm::errs() << "  " << *llvm::GetUnderlyingObject(SI->getPointerOperand(), F.getParent()->getDataLayout()) << "\n";
      //   SmallVector<const Value *, 4> Objects;
      //   llvm::GetUnderlyingObjects(SI->getPointerOperand(), Objects, F.getParent()->getDataLayout());
      //   llvm::errs() << '\t' << "POINTER:\n";
      //   for ( auto *obj : Objects)
      //     llvm::errs() << '\t' << *obj << '\n';
      //   Objects.clear();
      //   llvm::errs() << '\t' << "VALUE:\n";
      //   llvm::GetUnderlyingObjects(SI->getValueOperand(), Objects, F.getParent()->getDataLayout());
      //   for ( auto *obj : Objects)
      //     llvm::errs() << '\t' << *obj << '\n';
      // }
      // else if ( auto *AI = dyn_cast<AllocaInst>(&I)) {
      //   llvm::errs() << ">>> Alloca Object: " <<  I << "\n";
      //   // llvm::errs() << "  " << *llvm::GetUnderlyingObject(AI, F.getParent()->getDataLayout()) << "\n";
      //   SmallVector<const Value *, 4> Objects;
      //   llvm::GetUnderlyingObjects(AI, Objects, F.getParent()->getDataLayout());
      //   for ( auto *obj : Objects)
      //     llvm::errs() << '\t' << *obj << '\n';
      // }
      // todo atomics
  //   }
  // }
// }

// void SymbolizerPass::getAnalysisUsage(AnalysisUsage &AU) const  {
//   AU.addRequired<LoopInfoWrapperPass>();
//   AU.setPreservesAll();
// }

// static const std::string INDEX_SUFFIX = "[]";
// static const std::string FIELD_SUFFIX = ".";

// static Value *getInterestingPtr(Instruction *I) {
//   if ( !I) return nullptr;

//   Value *Ptr = nullptr; // Find symbol for this

//   if ( auto *LI = dyn_cast<LoadInst>(I)) {
//     assert( isa<GetElementPtrInst>(LI->getPointerOperand()) && "Load with non-Gep ptr. Run Canonicalizer");
//     Ptr = LI->getPointerOperand();
//   }
//   else if ( auto *SI = dyn_cast<StoreInst>(I)) {
//     assert( isa<GetElementPtrInst>(SI->getPointerOperand()) && "Store with non-Gep ptr. Run Canonicalizer");
//     Ptr = SI->getPointerOperand();
//   }
//   else if ( auto *Atom = dyn_cast<AtomicRMWInst>(I)) {
//     assert( isa<GetElementPtrInst>(Atom->getPointerOperand()) && "Atomic with non-Gep ptr. Run Canonicalizer");
//     Ptr = Atom->getPointerOperand();
//   }
//   else if ( auto *CI = dyn_cast<CallInst>(I)) {
//     if ( nvvm::isNVVMAtomic(*CI->getCalledFunction()))
//       Ptr = CI->getArgOperand(0);
//   }
//   else if ( auto *MemCpy = dyn_cast<MemCpyInst>(I)) {
//     Ptr = MemCpy->getSource();
//   }
//   else if ( auto *MemSet = dyn_cast<MemSetInst>(I)) {
//     Ptr = MemSet->getDest();
//   }

//   return Ptr;
// }

// static Symbol getUnknownSymbol(Value *V) {
//   if ( auto *Arg = dyn_cast<Argument>(V))
//     return !Arg->getName().empty()? Arg->getName().str()
//                                   : (std::string("arg") + std::to_string(Arg->getArgNo()));
//   if ( auto *Alloca = dyn_cast<AllocaInst>(V))
//     return !Allocal->getName().empty()? Arg->getName().str()
//                                   : (std::string("arg") + std::to_string(Arg->getArgNo()));
// }

// static std::string getUnknownSymbol() {
//   static unsigned int UnknownCount = 0;
//   return std::string("ukwn.") + std::to_string(UnknownCount++);
// }

// static bool isStructByVal(Argument *Arg) {
//   return Arg->getType()->isPointerTy() && (dyn_cast<PointerType>(Arg->getType())->getElementType() && Arg->hasAttribute(Attribute::ByVal));
// }

// // static std::string getArgAccessSymbol(Argument *Arg, Value *Ptr) {

// //   if ( isStructByVal(Arg))

// // }

// static std::vector<StructSymbol> getSymbolsForStruct(StructType *Ty) {

// }

// static void createStructSymbols(Module &M, std::vector<Function*>& Kernels, StructSymbolMap &StructSymbols) {

//   for ( auto *Kernel : Kernels) {

//     /// 1. Check kernel arguments
//     for ( auto &Arg : Kernel->args()) {
//       StructType *StructTy;
//       /// Struct argument. This is usually materialized as
//       /// a pointer to a struct with ByVal attribute and
//       /// thus handled by the next case.
//       if ( Arg.getType()->isStructTy())
//         StructTy = dyn_cast<StructType>(Arg.getType());
//       /// Pointer to struct or struct passed as byval pointer
//       else if ( Arg.getType()->isPointerTy()
//                 && dyn_cast<PointerType>(Arg.getType())->getElementType()->isStructTy())
//         StructTy = dyn_cast<StructType>(dyn_cast<PointerType>(Arg.getType())->getElementType());
//       if ( StructSymbols.find(StructTy) == StructSymbols.end())
//         StructSymbols[StructTy] = getSymbolsForStruct(StructTy);
//     }
//   }
// }

namespace kerma {

const std::string& Namer::None = "";

static std::map<llvm::GlobalVariable *, std::string> GlobalMap;
static std::map<llvm::AllocaInst *, std::string> LocalMap;
static std::map<llvm::Argument *, std::string> ArgMap;

const std::string& Namer::GetNameForGlobal(GlobalVariable *GV, bool Gen) {
  static unsigned int GlobalCounter = 0;

  if ( !GV) return None;

  if ( auto Lookup = GlobalMap.find(GV);
            Lookup != GlobalMap.end())
      return Lookup->second;

  if ( auto *MD = findMDForGlobal(GV))
    GlobalMap[GV] = MD->getName().str();
  else
    GlobalMap[GV] = Gen? ("glb" + std::to_string(GlobalCounter++)) : None;

  return GlobalMap[GV];
}

const std::string& Namer::GetNameForAlloca(AllocaInst *Alloca, bool Gen) {
  static unsigned int AllocaCounter = 0;

  if ( !Alloca) return None;

  if ( auto Lookup = LocalMap.find(Alloca);
            Lookup != LocalMap.end())
    return Lookup->second;

  if ( auto *MD = findMDForAlloca(Alloca))
    LocalMap[Alloca] = MD->getName().str();
  else
    LocalMap[Alloca] = Gen? ("loc" + std::to_string(AllocaCounter++)) : None;

  return LocalMap[Alloca];
}

const std::string& Namer::GetNameForArg(Argument *Arg, bool Gen) {
  if ( !Arg) return None;

  if ( auto Lookup = ArgMap.find(Arg);
            Lookup != ArgMap.end())
      return Lookup->second;

  if ( auto *MD = findMDForArgument(Arg))
    ArgMap[Arg] = MD->getName().str();
  else
    ArgMap[Arg] = Gen? ("arg" + std::to_string(Arg->getArgNo())) : None;

  return ArgMap[Arg];
}

} // namespace kerma
