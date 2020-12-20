#ifndef KERMA_BASE_IF_STMT_H
#define KERMA_BASE_IF_STMT_H

#include "kerma/Analysis/DataDependence.h"
#include "kerma/Base/Stmt.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm-10/llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>

namespace kerma {

class If : public KermaNode {
private:
  unsigned ID;
  std::vector<llvm::BranchInst *> BranchInstructions;
  llvm::BasicBlock *ThenBB=nullptr;
  llvm::BasicBlock *ElseBB=nullptr;
  Stmt *ConditionStmt = nullptr;
  std::vector<KermaNode *> Then;
  std::vector<KermaNode *> Else;
  SourceRange CondRange;
  SourceRange ThenRange;
  SourceRange ElseRange;
  bool DataDep = false;
  bool TransDataDep = false;

public:
  If() = delete;
  If(const SourceRange &Range, KermaNode *Parent = nullptr);

  std::vector<llvm::BranchInst *> &getBranchInstructions() {
    return BranchInstructions;
  }

  void setThenBB(llvm::BasicBlock *BB) { ThenBB = BB; }
  void setElseBB(llvm::BasicBlock *BB) { ElseBB = BB; }
  llvm::BasicBlock *getThenBB() { return ThenBB; }
  llvm::BasicBlock *getElseBB() { return ElseBB; }

  bool isDataDependent() const override { return DataDep; }
  void setDataDependent(bool b) override { DataDep = b; }

  bool isTransitivelyDataDependent() const override { return TransDataDep; }
  void setTransitivelyDataDependent(bool b) override { TransDataDep = b; }

  void addBranchInstruction(llvm::BranchInst *BI) {
    if (BI) {
      BranchInstructions.push_back(BI);
      // DataDep |= IsPossibleDataDependentValue(BI);
    }
  }

  unsigned getNumBranchInstructions() { return BranchInstructions.size(); }

  void setCond(Stmt *Stmt) {
    // llvm::errs() << "SETTING " << Stmt->getRange() << " for " << getRange()
    // << '\n';
    if (!Stmt)
      llvm::errs() << "NULL COND for: " << this->getID() << '\n';
    this->ConditionStmt = Stmt;
    Stmt->setParent(this);
    // CondRange = Stmt.getRange();
  }

  Stmt *getConditionStmt() { return ConditionStmt; }
  const std::vector<KermaNode *> getThen() { return Then; }
  const std::vector<KermaNode *> getElse() { return Else; }

  void addThenChild(KermaNode *Child) {
    if (Child) {
      Child->setParent(this);
      Then.push_back(Child);
    }
  }

  void addElseChild(KermaNode *Child) {
    if (Child) {
      Child->setParent(this);
      Else.push_back(Child);
    }
  }

  virtual If &operator=(const If &O) {
    KermaNode::operator=(O);
    BranchInstructions = O.BranchInstructions;
    ThenBB = O.ThenBB;
    ElseBB = O.ElseBB;
    ConditionStmt = O.ConditionStmt;
    Then = O.Then;
    Else = O.Else;
    CondRange = O.CondRange;
    ThenRange = O.ThenRange;
    ElseRange = O.ElseRange;
    DataDep = O.DataDep;
    TransDataDep = O.TransDataDep;
    return *this;
  }

  static bool classof(const KermaNode *S);

  void setCondRange(const SourceRange &R) { CondRange = R; }
  void setThenRange(const SourceRange &R) { ThenRange = R; }
  void setElseRange(const SourceRange &R) { ElseRange = R; }
  const SourceRange &getThenRange() const { return ThenRange; }
  const SourceRange &getElseRange() const { return ElseRange; }
  const SourceRange &getCondRange() const { return CondRange; }

  virtual void print(llvm::raw_ostream &O) const override {
    O << std::string(getNesting(), '\t') << "(IF) " << (DataDep ? "+" : "-")
      << (TransDataDep ? "+ " : "- ") << getRange() << " #" << getID() << " >"
      << getNesting() << " then: " << Then.size() << ", else: " << Else.size();
    if (getParent()) {
      O << ", parent: #" << getParent()->getID();
    } else {
      O << ", parent: none";
    }
    O << '\n' << std::string(getNesting() + 1, '\t') << "[cond: " << CondRange;
    if (ConditionStmt) {
      O << ' ' << BranchInstructions.size() << "x ]\n" << *(this->ConditionStmt);
    } else {
      O << ']';
    }
    O << '\n'
      << std::string(getNesting() + 1, '\t') << "[then: " << ThenRange << ']';
    for (auto *ThenChild : Then)
      O << '\n' << *ThenChild;
    if (getElseRange()) {
      O << "\n"
        << std::string(getNesting() + 1, '\t') << "[else: " << ElseRange << ']';
      for (auto *ElseChild : Else)
        O << '\n' << *ElseChild;
    }
  }
};

} // namespace kerma

#endif // KERMA_BASE_IF_STMT_H