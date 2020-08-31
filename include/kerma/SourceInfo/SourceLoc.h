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
  virtual ~SourceLoc()=default;

  virtual int getLine() const;
  virtual int getCol() const;
  virtual SourceLoc& setLine(int line);
  virtual SourceLoc& setCol(int col);
  virtual SourceLoc& set(int line, int col);

  virtual SourceLoc& invalidate();
  virtual bool isValid() const;

  virtual operator bool() const;
  virtual bool operator==(const SourceLoc &other) const;
  virtual bool operator!=(const SourceLoc &other) const;
  virtual bool operator<(const SourceLoc &other) const;
  virtual bool operator<=(const SourceLoc &other) const;
  virtual bool operator>(const SourceLoc &other) const;
  virtual bool operator>=(const SourceLoc &other) const;

  friend std::ostream & operator<<(std::ostream &os, const SourceLoc &loc);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, SourceLoc &loc);

  static const SourceLoc Unknown;

};

}

#endif // KERMA_SOURCEINFO_SOURCELOC_H