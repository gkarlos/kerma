#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>

using namespace kerma;
using namespace llvm;

char BreakConstantGEPPass::ID = 110;

BreakConstantGEPPass::BreakConstantGEPPass() : FunctionPass(ID) {}

static ConstantExpr * hasConstantGEP (Value * V) {
  if ( !V) return nullptr;
  if (ConstantExpr * CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return CE;
    for (unsigned index = 0; index < CE->getNumOperands(); ++index)
      if (hasConstantGEP (CE->getOperand(index)))
        return CE;
  }
  return nullptr;
}

bool BreakConstantGEPPass::runOnFunction(Function &F) {
  bool changed = false;
  SmallVector<Instruction*, 32> Worklist;

  for ( auto& BB : F)
    for ( auto& I : BB)
      if ( hasConstantGEP(&I))
        Worklist.push_back(&I);

  changed = Worklist.size();

  while ( Worklist.size()) {
    Instruction *I = Worklist.pop_back_val();

    if (auto* PHI = dyn_cast<PHINode>(I)) {
      for (unsigned index = 0; index < PHI->getNumIncomingValues(); ++index) {
        //
        // For PHI Nodes, if an operand is a constant expression with a GEP, we
        // want to insert the new instructions in the predecessor basic block.
        //
        // Note: It seems that it's possible for a phi to have the same
        // incoming basic block listed multiple times; this seems okay as long
        // the same value is listed for the incoming block.
        //
        Instruction * InsertPt = PHI->getIncomingBlock(index)->getTerminator();

        if (ConstantExpr * CE = hasConstantGEP (PHI->getIncomingValue(index))) {
          Instruction * NewInst = CE->getAsInstruction();

          for ( unsigned i = index; i < PHI->getNumIncomingValues(); ++i)
            if ( PHI->getIncomingBlock(i) == PHI->getIncomingBlock(index))
              PHI->setIncomingValue(i, NewInst);

          Worklist.push_back(NewInst);
        }
      }
    } else {
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        //
        // For other instructions, we want to insert instructions replacing
        // constant expressions immediently before the instruction using the
        // constant expression.
        //
        if (ConstantExpr * CE = hasConstantGEP (I->getOperand(index))) {
          Instruction * NewInst = CE->getAsInstruction();
          I->replaceUsesOfWith(CE, NewInst);
          Worklist.push_back(NewInst);
        }
      }
    }
  }

  return changed;
}

namespace {
static RegisterPass<BreakConstantGEPPass> RegisterBreakConstantGEPPass (
        /* pass arg  */   "kerma-break-gep",
        /* pass name */   "Replace constant GEP with GEP instruction",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}