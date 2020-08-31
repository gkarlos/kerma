#include "kerma/SourceInfo/SourceLoc.h"
#include <llvm/Support/raw_ostream.h>
#include <ostream>

namespace kerma {

static unsigned int InvalidVal = std::numeric_limits<int>::max();

const SourceLoc SourceLoc::Unknown( InvalidVal, InvalidVal);

SourceLoc::SourceLoc(unsigned int line, unsigned int col)
: L(line), C(col)
{}

SourceLoc::SourceLoc(const SourceLoc& other) 
: L(other.L), C(other.C)
{}

SourceLoc::SourceLoc(SourceLoc&& other)
: L(std::move(other.L)), C(std::move(other.C))
{}

unsigned int SourceLoc::getCol() const { return C; }

unsigned int SourceLoc::getLine() const { return L; }

SourceLoc& SourceLoc::setLine(unsigned int line) {
  L = line;
  if ( L == InvalidVal)
    C = L;
  return *this;
}

SourceLoc& SourceLoc::setCol(unsigned int col) {
  C = col;
  if ( C == InvalidVal)
    L = C;
  return *this;
}

SourceLoc& SourceLoc::set(unsigned int line, unsigned int col) {
  L = line;
  C = col;
  if ( L == InvalidVal || C == InvalidVal)
    L = C = InvalidVal;
  return *this;
}

SourceLoc& SourceLoc::invalidate() {
  L = InvalidVal;
  C = InvalidVal;
  return *this;
}

bool SourceLoc::isValid() const {
  // dont need to check C too. By construction they
  // are either both invalid or none is.
  return L != InvalidVal; 
}

bool SourceLoc::isInvalid() const {
  return !isValid();
}

SourceLoc::operator bool() const {
  return isValid();
}

SourceLoc& SourceLoc::operator=(const SourceLoc& other) {
  L = other.L;
  C = other.C;
  if ( L == InvalidVal || C == InvalidVal)
    L = C = InvalidVal;
  return *this;
}


bool SourceLoc::operator==(const SourceLoc &other) const {
  return L == other.L && C == other.C;
}

bool SourceLoc::operator!=(const SourceLoc &other) const {
  return !(*this == other);
}

bool SourceLoc::operator<(const SourceLoc &other) const {
  return L == other.L? C < other.C : L < other.L;
}

bool SourceLoc::operator<=(const SourceLoc &other) const {
  return L == other.L? C <= other.C : L <= other.L;
}

bool SourceLoc::operator>(const SourceLoc &other) const {
  return L == other.L? C > other.C : L > other.L;
}

bool SourceLoc::operator>=(const SourceLoc &other) const {
  return L == other.L? C >= other.C : L >= other.L;
}

std::ostream& operator<<(std::ostream& os, const SourceLoc& loc) {
  os << loc.L << ':' << loc.C;
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, SourceLoc& loc) {
  os << loc.L << ':' << loc.C;
  return os;
}

} // namespace kerma