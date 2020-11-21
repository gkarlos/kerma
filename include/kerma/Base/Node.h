#ifndef KERMA_BASE_NODE_H
#define KERMA_BASE_NODE_H

#include "kerma/SourceInfo/SourceRange.h"
#include <llvm/Support/raw_ostream.h>
#include <ostream>
#include <vector>

namespace kerma {

class KermaNode {
public:
  // We should probably make the Kernel/Function also a KermaNode
  // but for now we store kernel nodes in a hashmap with entries
  // <Kernel, NodeList>
  enum NodeKind { NK_MemStmt = 0, NK_Loop};

private:
  NodeKind Kind;

protected:
  KermaNode *Parent = nullptr;
  SourceRange Range;
  static unsigned genID();
  virtual void print(llvm::raw_ostream &O) {
    O << "(?) Node " << this;
  }

public:
  KermaNode()=delete;
  KermaNode(const KermaNode &O) { *this = O; }
  KermaNode(NodeKind Kind, const SourceRange &Range,
            KermaNode *Parent = nullptr)
      : Kind(Kind), Range(Range), Parent(Parent) {}
  NodeKind getKind() const { return Kind; }
  virtual const SourceRange &getRange() const { return Range; }
  virtual KermaNode *getParent() const { return Parent; }
  virtual void setParent(KermaNode *Node) { Parent = Node; }
  virtual unsigned getNesting() const {
    return Parent ? (1 + Parent->getNesting()) : 0;
  }

  virtual KermaNode &operator=(const KermaNode &O) {
    Parent = O.Parent;
    Range = O.Range;
    return *this;
  }
  virtual bool operator==(const KermaNode &O) const {
    return Range == O.Range;
  }
  virtual bool operator!=(const KermaNode &O) const {
    return !operator==(O);
  }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &O, KermaNode &KN) {
    KN.print(O);
    return O;
  }
};

} // namespace kerma

#endif // KERMA_BASE_NODE_H
