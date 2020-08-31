#include "kerma/SourceInfo/SourceRange.h"
#include "kerma/SourceInfo/SourceLoc.h"

namespace kerma {

SourceRange::SourceRange()=default;

SourceRange::SourceRange(SourceLoc start)
: S(start), E(start)
{}

SourceRange::SourceRange(SourceLoc start, SourceLoc end)
: S(start), E(end)
{}

SourceLoc & SourceRange::getStart() { return S; }

SourceLoc & SourceRange::getEnd() { return E; }

bool SourceRange::isValid() const { return S.isValid(); }

bool SourceRange::isInvalid() const { return S.isInvalid(); }

bool SourceRange::operator==(const SourceRange &other) const {
  return  S == other.S && E == other.E;
}

bool SourceRange::operator!=(const SourceRange &other) const {
  return !(*this == other);
}

std::ostream & operator<<(std::ostream &os, const SourceRange &loc) {
  os << loc.S << "," << loc.E;
  return os;
}

llvm::raw_ostream & operator<<(llvm::raw_ostream &os, SourceRange &loc) {
  os << loc.S << "," << loc.E;
  return os;
}

} // namespace kerma