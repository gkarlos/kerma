#ifndef KERMA_SOURCEINFO_SOURCELOC_H
#define KERMA_SOURCEINFO_SOURCELOC_H

#include <llvm/Support/raw_ostream.h>

#include <ostream>

namespace kerma {

class SourceLoc {
private:
  unsigned int L;
  unsigned int C;

public:
  SourceLoc(unsigned int line=0, unsigned int col=0);
  SourceLoc(const SourceLoc& other);
  SourceLoc(SourceLoc&& other);
  ~SourceLoc()=default;

  unsigned int getLine() const;
  unsigned int getCol() const;

  /// Set the line and column of this location
  /// If any of the values becomes numeric_limits<int>::max()
  /// then the other will do so too. In other words, setting
  /// any value (line or col) to numeric_limits<int>::max()
  /// will invalidate the location
  SourceLoc& set(unsigned int line, unsigned int col);
  SourceLoc& setLine(unsigned int line);
  SourceLoc& setCol(unsigned int col);

  /// Check if this is a valid location
  ///
  /// Invalid locations are used to represent the absense of a 
  /// location. For instance if the End of a SourceRange is 
  /// invalid, then the End location is the end of the file
  bool isValid() const;
  bool isInvalid() const;
  SourceLoc& invalidate();
  
  /// Bool operator returns true if the location is valid
  operator bool() const;

  SourceLoc& operator=(const SourceLoc &other);

  bool operator==(const SourceLoc &other) const;
  bool operator!=(const SourceLoc &other) const;

  bool operator<(const SourceLoc &other) const;
  bool operator<=(const SourceLoc &other) const;
  bool operator>(const SourceLoc &other) const;
  bool operator>=(const SourceLoc &other) const;

  friend std::ostream & operator<<(std::ostream &os, const SourceLoc &loc);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, SourceLoc &loc);

  /// An invalid location to denote the
  /// absence of source location info
  static const SourceLoc Unknown;
};

}

#endif // KERMA_SOURCEINFO_SOURCELOC_H