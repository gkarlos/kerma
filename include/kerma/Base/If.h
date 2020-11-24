#ifndef KERMA_BASE_IF_STMT_H
#define KERMA_BASE_IF_STMT_H

#include "kerma/Base/Stmt.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/IR/Instructions.h>

namespace kerma {

class If : public KermaNode {
public:
  If() = delete;
  If(const SourceRange &Range, KermaNode *Parent = nullptr);
  // If(const If& O) : If(O.Range, O.Parent) {
  //   *this = O;
  // }
  const std::vector<llvm::BranchInst *> getConditionValues() {
    return ConditionValues;
  }

  void addConditionValue(llvm::BranchInst *BI) {
    if (BI)
      ConditionValues.push_back(BI);
  }

  void setCond(Stmt *Stmt) {
      // llvm::errs() << "SETTING " << Stmt->getRange() << " for " << getRange() << '\n';
    if ( !Stmt)
      llvm::errs() << "NULL COND for: " << this->getID() << '\n';
    this->Condition = Stmt;
    Stmt->setParent(this);
    // CondRange = Stmt.getRange();
  }

  const Stmt *getCondition() { return Condition; }
  const std::vector<KermaNode *> getThen() { return Then; }
  const std::vector<KermaNode *> getElse() { return Else; }

  void addThenChild(KermaNode *Child) {
    if ( Child) {
      Child->setParent(this);
      Then.push_back(Child);
    }
  }

  void addElseChild(KermaNode *Child) {
    if ( Child) {
      Child->setParent(this);
      Else.push_back(Child);
    }
  }

  virtual If &operator=(const If &O) {
    KermaNode::operator=(O);
    ConditionValues = O.ConditionValues;
    Condition = O.Condition;
    Then = O.Then;
    Else = O.Else;
    CondRange = O.CondRange;
    ThenRange = O.ThenRange;
    ElseRange = O.ElseRange;
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
    O << std::string(getNesting(), '\t') << "(IF) " << getRange() << " #"
      << getID() << " then: " << Then.size() << ", else: " << Else.size();
    if (getParent()) {
      O << " parent: #" << getParent()->getID();
    } else {
      O << " parent: none";
    }
    O << '\n' << std::string(getNesting() + 1, '\t') << "[cond: " << CondRange;
    if ( Condition) {
      O << ' ' << ConditionValues.size() << "x BranchInst]\n" << *(this->Condition);
    } else {
      O << ']';
    }
    O << '\n' << std::string(getNesting() + 1, '\t') << "[then: " << ThenRange << ']';
    for (auto *ThenChild : Then)
      O << '\n' << *ThenChild;
    if (getElseRange()) {
      O << "\n" << std::string(getNesting() + 1, '\t') << "[else: " << ElseRange << ']';
      for (auto *ElseChild : Else)
        O << '\n' << *ElseChild;
    }
  }

private:
  unsigned ID;
  std::vector<llvm::BranchInst *> ConditionValues;
  Stmt *Condition = nullptr;
  std::vector<KermaNode *> Then;
  std::vector<KermaNode *> Else;
  SourceRange CondRange;
  SourceRange ThenRange;
  SourceRange ElseRange;
};

} // namespace kerma

#endif // KERMA_BASE_IF_STMT_H