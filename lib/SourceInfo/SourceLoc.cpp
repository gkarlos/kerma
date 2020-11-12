#include "kerma/SourceInfo/SourceLoc.h"
#include "boost/container_hash/hash_fwd.hpp"
#include <llvm/Support/raw_ostream.h>
#include <ostream>

namespace kerma {

using namespace llvm;

const SourceLoc SourceLoc::Unknown(0, 0);

SourceLoc::SourceLoc(unsigned int line, unsigned int col) : L(line), C(col) {
  if (!L)
    C = 0;
}

SourceLoc::SourceLoc(const DebugLoc &DL)
    : SourceLoc(DL.getLine(), DL.getCol()) {}

SourceLoc::SourceLoc(const SourceLoc &other) : L(other.L), C(other.C) {}

SourceLoc::SourceLoc(SourceLoc &&other) : L(other.L), C(other.C) {}

SourceLoc &SourceLoc::setLine(unsigned int line) {
  L = line;
  if (!L)
    C = L;
  return *this;
}

SourceLoc &SourceLoc::setCol(unsigned int col) {
  C = col;
  return *this;
}

SourceLoc &SourceLoc::set(unsigned int line, unsigned int col) {
  L = line;
  C = col;
  if (!L)
    C = 0;
  return *this;
}

SourceLoc &SourceLoc::invalidate() {
  L = C = 0;
  return *this;
}

bool SourceLoc::isValid() const { return L; }

bool SourceLoc::isInvalid() const { return !isValid(); }

bool SourceLoc::isPrecise() const { return L && C; }

SourceLoc::operator bool() const { return isValid(); }

SourceLoc &SourceLoc::operator=(const SourceLoc &other) {
  L = other.L;
  C = other.C;
  if (!L || !C)
    L = C = 0;
  return *this;
}

bool SourceLoc::operator==(const SourceLoc &other) const {
  return L == other.L && C == other.C;
}

bool SourceLoc::operator!=(const SourceLoc &other) const {
  return !(*this == other);
}

bool SourceLoc::operator<(const SourceLoc &other) const {
  return L == other.L ? C < other.C : L < other.L;
}

bool SourceLoc::operator<=(const SourceLoc &other) const {
  return L == other.L ? C <= other.C : L <= other.L;
}

bool SourceLoc::operator>(const SourceLoc &other) const {
  return L == other.L ? C > other.C : L > other.L;
}

bool SourceLoc::operator>=(const SourceLoc &other) const {
  return L == other.L ? C >= other.C : L >= other.L;
}

std::ostream &operator<<(std::ostream &os, const SourceLoc &loc) {
  os << loc.L << ':' << loc.C;
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SourceLoc &loc) {
  os << loc.L << ':' << loc.C;
  return os;
}

SourceLoc SourceLoc::from(const llvm::DebugLoc &DL) {
  return SourceLoc(DL.getLine(), DL.getCol());
}

} // namespace kerma

std::size_t
std::hash<kerma::SourceLoc>::operator()(const kerma::SourceLoc &Loc) const {
  return (Loc.getLine() ^ (Loc.getCol() << 1)) >> 1;
}
