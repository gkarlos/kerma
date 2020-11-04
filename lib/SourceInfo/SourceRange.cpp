#include "kerma/SourceInfo/SourceRange.h"
#include "kerma/SourceInfo/SourceLoc.h"

namespace kerma {

SourceRange::SourceRange(SourceLoc start)
: S(start), E(start)
{}

SourceRange::SourceRange(SourceLoc start, SourceLoc end)
: S(start), E(end < start? start : end)
{}

bool SourceRange::isValid() const { return S.isValid(); }

bool SourceRange::isInvalid() const { return S.isInvalid(); }

bool SourceRange::isEmpty() const { return S == E; }

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

llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const SourceRange &loc) {
  os << loc.S << "," << loc.E;
  return os;
}

const SourceRange SourceRange::Unknown(SourceLoc::Unknown);

} // namespace kerma

std::size_t std::hash<kerma::SourceRange>::operator()(const kerma::SourceRange& Range) const {
  return (std::hash<kerma::SourceLoc>()(Range.getStart())
          ^ (std::hash<kerma::SourceLoc>()(Range.getEnd()) << 1)) >> 1;
}
