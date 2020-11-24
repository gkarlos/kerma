#include "kerma/Base/Node.h"

#include <mutex>

namespace kerma {

static std::mutex mtx;

unsigned genID() {
  static unsigned int IDs = 0;
  unsigned int id;
  mtx.lock();
  id = IDs++;
  mtx.unlock();
  return id;
}

static unsigned id = 0;

KermaNode::KermaNode(NodeKind Kind, const SourceRange &Range, KermaNode *Parent)
    : KermaNode(genID(), Kind, Range, Parent) {}

// KermaNode::KermaNode(unsigned ID, NodeKind Kind, const SourceRange &Range,
//                      KermaNode *Parent)
//     : ID(ID), Kind(Kind), Range(Range), Parent(Parent) {}

} // namespace kerma