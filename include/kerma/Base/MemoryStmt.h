#ifndef KERMA_BASE_MEMORY_STMT
#define KERMA_BASE_MEMORY_STMT

#include "kerma/Base/Node.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <ostream>

namespace kerma {

// Represents a source code statement that contains
// a number of MemoryAccesses (at least 1)
class MemoryStmt : public KermaNode {
public:
  enum Type : unsigned { UKN = 0, RD, WR, RDWR };

protected:
  MemoryStmt(unsigned ID, SourceRange R, Type Ty, KermaNode *Parent);

public:
  MemoryStmt() : MemoryStmt(SourceRange::Unknown) {}
  MemoryStmt(SourceRange R) : MemoryStmt(R, UKN) {}
  MemoryStmt(SourceRange R, Type Ty) : MemoryStmt(R, Ty, nullptr) {}
  MemoryStmt(SourceRange R, Type Ty, KermaNode *Parent);

  const Type getType() const { return Ty; }
  const unsigned getID() const { return ID; }

  // Set the range of the statement. If the statement
  // already contains a number of accesses, then those
  // accesses that do not fit the new range are removed
  // Setting to SourceRange::Unknown is equivalent to
  // removing all MemoryAccesses
  MemoryStmt &setRange(const SourceRange &R);

  virtual MemoryStmt &operator=(const MemoryStmt &O) {
    KermaNode::operator=(O);
    Ty = O.Ty;
    MAS = O.MAS;
    return *this;
  }
  // Add a MemoryAccess to this statement
  // If the MemoryStmt already has a known range and \p MI does not
  // fit that range, false is returned <br/>
  // If the MemoryStmt is has an unknown range, the MemoryStmts range becomes
  // becomes the range of the statement \p MI belongs to which is
  // looked up in \p SI. <br> If the lookup fails \p MI is not added
  bool addMemoryAccess(MemoryAccess &MI, SourceInfo &SI);


  const std::vector<MemoryAccess> & getAccesses() const { return MAS; }
  const unsigned int getNumAccesses() const { return MAS.size(); }
  virtual void print(llvm::raw_ostream &O) const override;
  // friend std::ostream & operator<<(std::ostream &os, const MemoryStmt &S);
  // friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const MemoryStmt &S);
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const MemoryStmt &KN) {
    KN.print(O);
    return O;
  }

  static bool classof(const KermaNode *S);

private:
  KermaNode *Parent;
  unsigned ID;
  std::vector<MemoryAccess> MAS;
  SourceRange R;
  Type Ty;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_STMT