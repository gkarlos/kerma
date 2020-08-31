#include "kerma/SourceInfo/SourceLoc.h"
#include <llvm/Support/raw_ostream.h>
#include <ostream>

namespace kerma {

static int INVALID_LINE = -1;
static int INVALID_COL = -1;

SourceLoc::SourceLoc(int line, int col)
: line(line), col(col)
{}

SourceLoc::SourceLoc(const SourceLoc& other) 
: line(other.line), col(other.col)
{}

SourceLoc::SourceLoc(SourceLoc&& other)
: line(std::move(other.line)), col(std::move(other.col))
{}

int SourceLoc::getCol() const { return this->col; }

int SourceLoc::getLine() const { return this->line; }

SourceLoc& SourceLoc::setLine(int line) {
  this->line = line;
  return *this;
}

SourceLoc& SourceLoc::setCol(int col) {
  this->col = col;
  return *this;
}

SourceLoc& SourceLoc::set(int line, int col) {
  this->line = line;
  this->col = col;
  return *this;
}

SourceLoc& SourceLoc::invalidate() {
  this->line = INVALID_LINE;
  this->col = INVALID_COL;
  return *this;
}

bool SourceLoc::isValid() const {
  return this->line >= 0 && this->col >= 0;
}

SourceLoc::operator bool() const {
  return this->isValid();
}

bool SourceLoc::operator==(const SourceLoc &other) const {
  return this->line == other.line && this->col == other.col;
}

bool SourceLoc::operator!=(const SourceLoc &other) const {
  return !(*this == other);
}

bool SourceLoc::operator<(const SourceLoc &other) const {
  return this->line == other.line? this->col < other.col : this->line < other.line;
}

bool SourceLoc::operator<=(const SourceLoc &other) const {
  return (*this < other) || (*this == other);
}

bool SourceLoc::operator>(const SourceLoc &other) const {
  return this->line == other.line? this->col > other.col : this->line > other.line;
}

bool SourceLoc::operator>=(const SourceLoc &other) const {
  return (*this > other) || (*this == other);
}

std::ostream& operator<<(std::ostream& os, const SourceLoc& loc) {
  os << loc.line << ',' << loc.col;
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, SourceLoc& loc) {
  os << loc.line << ',' << loc.col;
  return os;
}



} // namespace kerma