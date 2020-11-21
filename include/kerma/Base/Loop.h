#ifndef KERMA_BASE_LOOP_NEST_H
#define KERMA_BASE_LOOP_NEST_H

#include "kerma/Base/Node.h"
#include <llvm/Support/raw_ostream.h>
#include <llvm/Analysis/LoopInfo.h>
#include <memory>

namespace kerma {

class LoopNest : public KermaNode {
protected:
  LoopNest(unsigned ID, llvm::Loop *Loop, KermaNode *Parent);

public:
  LoopNest()=delete;
  LoopNest(llvm::Loop *Loop) : LoopNest(Loop, nullptr) {}
  LoopNest(llvm::Loop *Loop, KermaNode *Parent);
  LoopNest(const LoopNest &O) : LoopNest(O.L, O.Parent) {}

public:
  llvm::Loop *getLoop() { return L; }
  void addChild(KermaNode &Child);
  void addChild(KermaNode &&Child);
  std::vector<KermaNode*> getLoopChildren();
  std::vector<KermaNode*> getNonLoopChildren();
  const std::vector<KermaNode*> &getChildren() { return Children; }
  unsigned getID() { return ID; }

  virtual void print(llvm::raw_ostream &O) override {
    O << std::string(getNesting(), '\t') << "(Loop) " << getRange() << " #" << getID() << " children: " << Children.size() << ", ";
    if ( getParent()) {
      O << " parent: #" << llvm::dyn_cast<LoopNest>(getParent())->getID();
    } else {
      O << " parent: none";
    }
    for (auto *Child : Children)
      O << '\n' << *Child;
  }


  virtual LoopNest &operator=(const LoopNest &O) {
    KermaNode::operator=(O);
    L = O.L;
    Children = O.Children;
    return *this;
  }

  static bool classof(const KermaNode *S);

private:
  unsigned ID;
  llvm::Loop *L;
  std::vector<KermaNode *> Children;
};

}

#endif