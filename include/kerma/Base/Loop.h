#ifndef KERMA_BASE_LOOP_NEST_H
#define KERMA_BASE_LOOP_NEST_H

#include "kerma/Base/Node.h"
#include <llvm-10/llvm/IR/BasicBlock.h>
#include <llvm-10/llvm/IR/Instructions.h>
#include <llvm-10/llvm/IR/Metadata.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace kerma {

class LoopNest : public KermaNode {
private:
  bool DataDep = false;
  bool TransDataDep = false;
  std::vector<KermaNode *> InitChildren;
  std::vector<KermaNode *> Children;
  llvm::MDNode *LoopID = nullptr;
  llvm::PHINode *IV = nullptr;
  llvm::BranchInst *LoopGuard = nullptr;
  llvm::BasicBlock *LoopHeader = nullptr;
  llvm::BasicBlock *LoopPreheader = nullptr;
  llvm::BasicBlock *LoopLatch = nullptr;
  unsigned LoopDepth;

protected:
  LoopNest(unsigned ID, llvm::Loop *Loop, KermaNode *Parent);

public:
  LoopNest() = delete;
  LoopNest(llvm::Loop *Loop) : LoopNest(Loop, nullptr) {}
  LoopNest(llvm::Loop *Loop, KermaNode *Parent)
      : KermaNode(NK_Loop,
                  SourceRange(Loop->getLocRange().getStart(),
                              Loop->getLocRange().getEnd()),
                  Parent) {
    assert(Loop && "Loop is null!");
    LoopID = Loop->getLoopID();
    LoopHeader = Loop->getHeader();
    LoopLatch = Loop->getLoopLatch();
    LoopPreheader = Loop->getLoopPreheader();
    LoopGuard = Loop->getLoopGuardBranch();
    LoopDepth = Loop->getLoopDepth();
    // We cant set the induction variable here as it requires SCEV info
    // We do that in the MATBuilder instead
  }
  void setInductionVariable(llvm::PHINode *IV) { this->IV = IV; }
  void setHeader(llvm::BasicBlock *BB) { LoopHeader = BB; }
  void setPreheader(llvm::BasicBlock *BB) { LoopPreheader = BB; }
  void setLatch(llvm::BasicBlock *BB) { LoopLatch = BB; }
  void setGuard(llvm::BranchInst *BI) { LoopGuard = BI; }

  llvm::MDNode *getLoopID() { return LoopID; }
  llvm::PHINode *getInductionVariable() { return IV; }
  llvm::BranchInst *getGuard() { return LoopGuard; }
  llvm::BasicBlock *getHeader() { return LoopHeader; }
  llvm::BasicBlock *getPreheader() { return LoopPreheader; }
  llvm::BasicBlock *getLatch() { return LoopLatch; }
  unsigned getDepth() { return LoopDepth; }

  // LoopNest(const LoopNest &O) : {
  //   *this =
  // }

public:
  void addInitChild(KermaNode &InitChild) {
    InitChild.setParent(this);
    InitChildren.push_back(&InitChild);
  }

  void addChild(KermaNode &Child, bool init = false) {
    Child.setParent(this);
    if (init) {
      InitChildren.push_back(&Child);
    } else {
      Children.push_back(&Child);
    }
  }
  void addChild(KermaNode &&Child, bool init = false) {
    Child.setParent(this);
    if (init) {
      InitChildren.push_back(&Child);
    } else {
      Children.push_back(&Child);
    }
  }

  void setDataDependent(bool v) override { DataDep = v; }
  bool isDataDependent() const override { return DataDep; }
  void setTransitivelyDataDependent(bool b = true) override {
    TransDataDep = b;
  }
  bool isTransitivelyDataDependent() const override {
    return Parent && (Parent->isDataDependent() || Parent->isTransitivelyDataDependent());
  }

  const std::vector<KermaNode *> &getInitChildren() { return InitChildren; }
  // std::vector<KermaNode *> getLoopChildren();
  // std::vector<KermaNode *> getNonLoopChildren();
  const std::vector<KermaNode *> &getChildren() { return Children; }

  virtual void print(llvm::raw_ostream &O) const override {
    O << std::string(getNesting(), '\t') << "(LOOP) " << (DataDep ? "+" : "-")
      << (TransDataDep ? "+ " : "- ") << getRange() << " #" << getID() << " >"
      << getNesting() << " @" << LoopDepth << " . iv: "
      << (IV ? (IV->getName().empty() ? "??" : ("\'" + IV->getName() + "\'"))
             : "unknown");
    O << ", init: " << InitChildren.size() << ", children: " << Children.size();
    if (getParent()) {
      O << ", parent: #" << getParent()->getID();
    } else {
      O << ", parent: none";
    }
    O << '\n' << std::string(getNesting() + 1, '\t') << "[init]";
    for (auto *InitChild : InitChildren)
      O << '\n' << *InitChild;
    O << '\n' << std::string(getNesting() + 1, '\t') << "[body]";
    for (auto *Child : Children)
      O << '\n' << *Child;
  }

  virtual LoopNest &operator=(const LoopNest &O) {
    KermaNode::operator=(O);
    Children = O.Children;
    DataDep = O.DataDep;
    TransDataDep = O.TransDataDep;
    LoopDepth = O.LoopDepth;
    LoopID = O.LoopID;
    LoopGuard = O.LoopGuard;
    IV = O.IV;
    LoopHeader = O.LoopHeader;
    LoopPreheader = O.LoopPreheader;
    return *this;
  }

  virtual bool operator==(const LoopNest &O) const {
    KermaNode::operator==(O);
    return (this->LoopID == O.LoopID);
  }

  static bool classof(const KermaNode *S);
};

} // namespace kerma

#endif