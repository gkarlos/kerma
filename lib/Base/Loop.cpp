#include "kerma/Base/Loop.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <memory>

using namespace llvm;

namespace kerma {

LoopNest::LoopNest(Loop *Loop, KermaNode *Parent)
    : LoopNest(KermaNode::genID(), Loop, Parent) {}

LoopNest::LoopNest(unsigned ID, Loop *Loop, KermaNode *Parent)
    : KermaNode(NK_Loop,
                SourceRange(Loop->getLocRange().getStart(),
                            Loop->getLocRange().getEnd()),
                Parent),
      ID(ID), L(Loop) {
  assert(Loop && "Loop is null!");
}

void LoopNest::addChild(KermaNode &Child) {
  Child.setParent(this);
  Children.push_back(&Child);
}

void LoopNest::addChild(KermaNode &&Child) {
  Child.setParent(this);
  Children.push_back(&Child);
}

bool LoopNest::classof(const KermaNode *S) { return S->getKind() == NK_Loop; }

} // namespace kerma