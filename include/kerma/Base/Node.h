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
  enum NodeKind { NK_Stmt = 0, NK_If, NK_Loop };

private:
  unsigned ID;
  NodeKind Kind;

protected:
  KermaNode *Parent = nullptr;
  SourceRange Range;
  virtual void print(llvm::raw_ostream &O) const { O << "(?) Node " << this; }

private:
  KermaNode(unsigned ID, NodeKind Kind, const SourceRange &Range,
            KermaNode *Parent = nullptr)
      : ID(ID), Kind(Kind), Range(Range), Parent(Parent) {}

public:
  KermaNode() = delete;
  KermaNode(const KermaNode &O) { *this = O; }
  KermaNode(NodeKind Kind, const SourceRange &Range,
            KermaNode *Parent = nullptr);
  NodeKind getKind() const { return Kind; }
  unsigned getID() const { return ID; }
  virtual const SourceRange &getRange() const { return Range; }
  KermaNode *getParent() const { return Parent; }
  void setParent(KermaNode *Node) { Parent = Node; }
  unsigned getNesting() const {
    return Parent ? (1 + Parent->getNesting()) : 0;
  }

  KermaNode &operator=(const KermaNode &O) {
    Parent = O.Parent;
    Range = O.Range;
    Kind = O.Kind;
    ID = O.ID;
    return *this;
  }
  virtual bool operator==(const KermaNode &O) const { return Range == O.Range; }
  virtual bool operator!=(const KermaNode &O) const { return !operator==(O); }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &O, KermaNode &KN) {
    KN.print(O);
    return O;
  }
};

} // namespace kerma

#endif // KERMA_BASE_NODE_H
