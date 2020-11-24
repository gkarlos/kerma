#ifndef KERMA_BASE_MEMORY_STMT
#define KERMA_BASE_MEMORY_STMT

#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/Node.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <ostream>

namespace kerma {

// Represents a source code statement that contains
// a number of MemoryAccesses (at least 1)
class Stmt : public KermaNode {
public:
  enum Type : unsigned { UKN = 0, RD, WR, RDWR };

  // protected:
  //   Stmt(unsigned ID, SourceRange R, Type Ty, KermaNode *Parent);

public:
  Stmt() : Stmt(SourceRange::Unknown, UKN) {}
  Stmt(SourceRange R) : Stmt(R, UKN) {}
  Stmt(SourceRange R, Type Ty, KermaNode *Parent=nullptr)
      : KermaNode(NK_Stmt, R, Parent), Ty(Ty) {}

  const Type getType() const { return Ty; }

  // Set the range of the statement. If the statement
  // already contains a number of accesses, then those
  // accesses that do not fit the new range are removed
  // Setting to SourceRange::Unknown is equivalent to
  // removing all MemoryAccesses
  Stmt &setRange(const SourceRange &R);

  virtual Stmt &operator=(const Stmt &O) {
    KermaNode::operator=(O);
    Ty = O.Ty;
    Accesses = O.Accesses;
    return *this;
  }

  // Add a MemoryAccess to this statement
  // If the Stmt already has a known range and \p MI does not
  // fit that range, false is returned <br/>
  // If the Stmt is has an unknown range, the Stmts range becomes
  // becomes the range of the statement \p MI belongs to which is
  // looked up in \p SI. <br> If the lookup fails \p MI is not added
  bool addMemoryAccess(MemoryAccess &MI, SourceInfo &SI);

  // Returns true if the statement is a memory statement, that
  // is type is RD/WR or RDWR. False otherwise
  operator bool() const { return getType() != UKN; }

  const std::vector<MemoryAccess> &getAccesses() const { return Accesses; }
  const unsigned int getNumAccesses() const { return Accesses.size(); }

  virtual void print(llvm::raw_ostream &O) const override;
  // friend std::ostream & operator<<(std::ostream &os, const Stmt &S);
  // friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const Stmt
  // &S);
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const Stmt &KN) {
    KN.print(O);
    return O;
  }

  static bool classof(const KermaNode *S);

private:
  // KermaNode *Parent;
  // unsigned ID;
  std::vector<MemoryAccess> Accesses;
  SourceRange R;
  Type Ty;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_STMT