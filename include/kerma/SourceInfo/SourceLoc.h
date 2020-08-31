#ifndef KERMA_SOURCEINFO_SOURCELOC_H
#define KERMA_SOURCEINFO_SOURCELOC_H

#include <llvm/Support/raw_ostream.h>

#include <ostream>

namespace kerma {

class SourceLoc {
private:
  int line;
  int col;

public:
  SourceLoc(int line=0, int col=0);
  SourceLoc(const SourceLoc& other);
  SourceLoc(SourceLoc&& other);
  ~SourceLoc()=default;

  int getLine() const;
  int getCol() const;
  SourceLoc& setLine(int line);
  SourceLoc& setCol(int col);
  SourceLoc& set(int line, int col);

  SourceLoc& invalidate();
  bool isValid() const;

  operator bool() const;
  bool operator==(const SourceLoc &other) const;
  bool operator!=(const SourceLoc &other) const;
  bool operator<(const SourceLoc &other) const;
  bool operator<=(const SourceLoc &other) const;
  bool operator>(const SourceLoc &other) const;
  bool operator>=(const SourceLoc &other) const;

  friend std::ostream & operator<<(std::ostream &os, const SourceLoc &loc);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, SourceLoc &loc);

  static const SourceLoc Unknown;

};

}

#endif // KERMA_SOURCEINFO_SOURCELOC_H