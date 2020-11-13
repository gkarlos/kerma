#ifndef KERMA_BASE_MEMORY_STMT
#define KERMA_BASE_MEMORY_STMT

#include "kerma/Base/MemoryAccess.h"
#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"

namespace kerma {

// Represents a source code statement that contains
// a number of MemoryAccesses (at least 1)
class MemoryStmt {
public:
  enum Type : unsigned { UKN = 0, RD, WR, RDWR };

protected:
  MemoryStmt(unsigned ID, SourceRange R, Type Ty);

public:
  MemoryStmt() : MemoryStmt(SourceRange::Unknown) {}
  MemoryStmt(SourceRange R) : MemoryStmt(R, UKN) {}
  MemoryStmt(SourceRange R, Type Ty);

  const Type getType() const { return Ty; }
  const unsigned getID() const { return ID; }

  // Set the range of the statement. If the statement
  // already contains a number of accesses, then those
  // accesses that do not fit the new range are removed
  // Setting to SourceRange::Unknown is equivalent to
  // removing all MemoryAccesses
  MemoryStmt &setRange(const SourceRange &R);
  const SourceRange &getRange() const { return R; }

  bool operator==(MemoryStmt &O) { return ID == O.ID; }

  // Add a MemoryAccess to this statement
  // If the stmt already has a known range and \p MI does not
  // fit that range, false is returned <br/>
  // If the stmt is has an unknown range, the stmts range becomes
  // becomes the range of the statement \p MI belongs to which is
  // looked up in \p SI. <br> If the lookup fails \p MI is not added
  bool addMemoryAccess(MemoryAccess &MI, SourceInfo &SI);


  const std::vector<MemoryAccess> & getAccesses() const { return MAS; }
  const unsigned int getNumAccesses() const { return MAS.size(); }

  friend std::ostream & operator<<(std::ostream &os, const MemoryStmt &S);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const MemoryStmt &S);


private:
  unsigned ID;
  std::vector<MemoryAccess> MAS;
  SourceRange R;
  Type Ty;
};

} // namespace kerma

#endif // KERMA_BASE_MEMORY_STMT