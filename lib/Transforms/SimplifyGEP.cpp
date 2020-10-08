#include "kerma/Transforms/SimplifyGEP.h"
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/Pass.h>
#include <llvm/PassSupport.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Local.h>

using namespace llvm;

/// https://github.com/llvm/llvm-project/blob/6bad3caeb079748a25fd34bd21255824c8dcb8f1/llvm/lib/Transforms/InstCombine/InstructionCombining.cpp#L3554

namespace kerma {

static bool shouldMergeGEPs(GEPOperator &GEP, GEPOperator &Src) {
  // If this GEP has only 0 indices, it is the same pointer as
  // Src. If Src is not a trivial GEP too, don't combine
  // the indices.
  if (GEP.hasAllZeroIndices() && !Src.hasAllZeroIndices() &&
      !Src.hasOneUse())
    return false;
  return true;
}

char SimplifyGEPPass::ID = 1;
const char * SimplifyGEPPass::PASS_NAME = "Kerma: Simplify GEP instructions/operands";

SimplifyGEPPass::SimplifyGEPPass() : FunctionPass(ID) {}

// Adapted from https://llvm.org/doxygen/InstructionCombining_8cpp_source.html
bool SimplifyGEPPass::simplifyGEP(llvm::GetElementPtrInst *GEP) {
  if ( !GEP) return false;

  Value *PtrOp = GEP->getOperand(0);

  if ( auto *Src = dyn_cast<GEPOperator>(PtrOp)) {

    if ( auto *SrcGEP = dyn_cast<GetElementPtrInst>(Src->getPointerOperand()))
      if ( SrcGEP->getNumOperands() == 2) {
        llvm::errs() << "kerma: SipmlifyGEP: this should not happen!";
        return false;
      }

    SmallVector<Value*, 8> indices;

    bool EndsWithSequential = false;
    for (gep_type_iterator I = gep_type_begin(*Src), E = gep_type_end(*Src); I != E; ++I)
      EndsWithSequential = !I.isStruct();

    if ( EndsWithSequential) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      Value *Sum;
      Value *SO1 = Src->getOperand(Src->getNumOperands()-1);
      Value *GO1 = GEP->getOperand(1);
      if (SO1 == Constant::getNullValue(SO1->getType())) {
        Sum = GO1;
      } else if (GO1 == Constant::getNullValue(GO1->getType())) {
        Sum = SO1;
      } else {
        // If they aren't the same type, then the input hasn't been processed
        // by the loop above yet (which canonicalizes sequential index types to
        // intptr_t).  Just avoid transforming this until the input has been
        // normalized.
        if (SO1->getType() != GO1->getType()) {
          return false;
        }
        Sum = llvm::BinaryOperator::Create(BinaryOperator::Add,
                                           SO1, GO1,
                                           PtrOp->getName() + ".sum" , GEP);
      }

      // Update the GEP in place if possible.
      if (Src->getNumOperands() == 2) {
        assert(isa<GetElementPtrInst>(Src) && "Inline GEP detected. Maybe NormConstantGEPs wasn't run!");

        GEP->setOperand(0, Src->getOperand(0));
        GEP->setOperand(1, Sum);
        DeleteSet.insert(cast<GetElementPtrInst>(Src));
        return true;
      }

      indices.append(Src->op_begin() + 1, Src->op_end() - 1);
      indices.push_back(Sum);
      indices.append(GEP->op_begin() + 2, GEP->op_end());
    }
    else if ( auto *Const = dyn_cast<Constant>(GEP->idx_begin());
              Const && Const->isNullValue() && Src->getNumOperands() != 1) {
      // Otherwise we can do the fold if the first index of the GEP is a zero

      indices.append(Src->op_begin() + 1, Src->op_end());
      indices.append(GEP->idx_begin() + 1, GEP->idx_end());
    }

    if ( !indices.empty()) {
      auto *GEPNew = (GEP->isInBounds() && Src->isInBounds())
        ? GetElementPtrInst::CreateInBounds(Src->getPointerOperand(), indices, GEP->getName()+".new", GEP)
        : GetElementPtrInst::Create(cast<PointerType>(Src->getPointerOperand()->getType())->getElementType(),
                                    Src->getPointerOperand(), indices, GEP->getName()+".new", GEP);
      GEPNew->setMetadata("dbg", GEP->getMetadata("dbg"));
      GEP->replaceAllUsesWith(GEPNew);

      // GEP is now dead. Mark for deletion
      DeleteSet.insert(GEP);
      return true;
    }
  }
  return false;
}

/// Poor mans DCE but only for GEP instructions
/// Returns true if something was deleted, and
/// otherwise false
static bool eliminateDeadGEPs(llvm::Function &F) {
  SmallSet<Instruction*, 32> DeleteSet;

  bool deletedSomething = false;

  for ( auto& BB : F)
    for ( auto& I : BB)
      if ( auto *GEP = dyn_cast<GetElementPtrInst>(&I))
        if ( I.getNumUses() == 0)
          DeleteSet.insert(&I);

  if ( !DeleteSet.empty()) {
    deletedSomething = true;
    for ( auto* I : DeleteSet)
      I->eraseFromParent();
  }

  return deletedSomething;
}

bool SimplifyGEPPass::runOnFunction(llvm::Function &F) {
  DeleteSet.clear();

  bool changed;

  int iter = 0;

  do {
    iter++;
    changed = false;
    for ( auto& BB : F)
      for ( auto& I : BB)
        if ( auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          changed |= simplifyGEP(GEP);

    // if ( !DeleteSet.empty()) {
    //   for ( auto *I : DeleteSet)
    //     I->eraseFromParent();
    //   DeleteSet.clear();
    //   changed = true;
    // }
    eliminateDeadGEPs(F);

  } while ( changed);

  llvm::errs() << this->getPassName() << " " << iter << " iterations" << "\n";

  return changed;
}

namespace {

static RegisterPass<SimplifyGEPPass> RegisterSimplifyGEPPass(
        /* arg      */ "kerma-simplify-gep",
        /* name     */ "Simplify GEP instructions/operands",
        /* CFGOnly  */ false,
        /* analysis */ true);

} // anonymous namespace

} // namespace kerma