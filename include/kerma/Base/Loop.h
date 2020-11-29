#ifndef KERMA_BASE_LOOP_NEST_H
#define KERMA_BASE_LOOP_NEST_H

#include "kerma/Base/Node.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace kerma {

class LoopNest : public KermaNode {
protected:
  LoopNest(unsigned ID, llvm::Loop *Loop, KermaNode *Parent);

public:
  LoopNest() = delete;
  LoopNest(llvm::Loop *Loop) : LoopNest(Loop, nullptr) {}
  LoopNest(llvm::Loop *Loop, KermaNode *Parent)
      : KermaNode(NK_Loop,
                  SourceRange(Loop->getLocRange().getStart(),
                              Loop->getLocRange().getEnd()),
                  Parent),
        L(Loop) {
    assert(Loop && "Loop is null!");
  }
  LoopNest(const LoopNest &O) : LoopNest(O.L, O.Parent) {}

public:
  llvm::Loop *getLoop() const { return L; }
  void addInitChild(KermaNode &InitChild) {
    InitChild.setParent(this);
    InitChildren.push_back(&InitChild);
  }

  void addChild(KermaNode &Child, bool init=false) {
    Child.setParent(this);
    if ( init) {
      InitChildren.push_back(&Child);
    } else {
      Children.push_back(&Child);
    }
  }
  void addChild(KermaNode &&Child, bool init=false) {
    Child.setParent(this);
    if ( init) {
      InitChildren.push_back(&Child);
    } else {
      Children.push_back(&Child);
    }
  }

  std::vector<KermaNode *> getInitChildren() { return InitChildren; }
  std::vector<KermaNode *> getLoopChildren();
  std::vector<KermaNode *> getNonLoopChildren();
  const std::vector<KermaNode *> &getChildren() { return Children; }

  virtual void print(llvm::raw_ostream &O) const override {
    O << std::string(getNesting(), '\t') << "(LOOP) " << getRange() << " #"
      << getID() << " init: " << InitChildren.size() << ", children: " << Children.size();
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
    L = O.L;
    Children = O.Children;
    return *this;
  }

  virtual bool operator==(const LoopNest &O) const {
    // KermaNode::operator==(O);
    return (this->L == O.getLoop());
  }

  static bool classof(const KermaNode *S);

private:
  llvm::Loop *L;
  std::vector<KermaNode *> InitChildren;
  std::vector<KermaNode *> Children;
};

} // namespace kerma

#endif