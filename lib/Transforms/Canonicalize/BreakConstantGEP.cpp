#include "kerma/Transforms/Canonicalize/BreakConstantGEP.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/WithColor.h>

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
    for ( auto& I : BB) {
      for (unsigned index = 0; index < I.getNumOperands(); ++index) {
        if (hasConstantGEP (I.getOperand(index))) {
          Worklist.push_back(&I);
        }
      }
    }

  changed = Worklist.size();

  while ( Worklist.size()) {
    Instruction *I = Worklist.pop_back_val();

    if (auto* PHI = dyn_cast<PHINode>(I)) {
      for (unsigned index = 0; index < PHI->getNumIncomingValues(); ++index) {
        // For PHI Nodes, if an operand is a constant expression with a GEP, we
        // want to insert the new instructions at the end of the incoming block.
        Instruction * InsertPt = PHI->getIncomingBlock(index)->getTerminator();

        if (ConstantExpr * CE = hasConstantGEP (PHI->getIncomingValue(index))) {

          Instruction * NewInst = CE->getAsInstruction();
          NewInst->insertBefore(InsertPt);
          NewInst->setName("brk.gep");

          for ( unsigned i = index; i < PHI->getNumIncomingValues(); ++i)
            if ( PHI->getIncomingBlock(i) == PHI->getIncomingBlock(index))
              PHI->setIncomingValue(i, NewInst);

          Worklist.push_back(NewInst);
          if ( CE->user_empty())
            CE->destroyConstant();
        }
      }
    } else {
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        // For other instructions, just insert the instruction replacing
        // the operand just before the instruction itself
        if (ConstantExpr * CE = hasConstantGEP (I->getOperand(index))) {

          Instruction * NewInst = CE->getAsInstruction();
          NewInst->insertBefore(I);
          NewInst->setName("brk.gep");

          I->replaceUsesOfWith(CE, NewInst);

          Worklist.push_back(NewInst);
          if ( CE->user_empty())
            CE->destroyConstant();
        }
      }
    }
  }

#ifdef KERMA_OPT_PLUGIN
  WithColor::note();
  WithColor(errs(), raw_ostream::Colors::CYAN) << "BreakConstantGEP: ";
  errs() << demangle(F.getName()) << ": " << changed << '\n';
#endif
  return changed;
}

namespace {
static RegisterPass<BreakConstantGEPPass> RegisterBreakConstantGEPPass (
        /* pass arg  */   "kerma-break-gep",
        /* pass name */   "Replace constant GEP with GEP instruction",
        /* modifies CFG */ false,
        /* analysis pass*/ false);
}