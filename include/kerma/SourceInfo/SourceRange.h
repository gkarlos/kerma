#ifndef KERMA_SOURCEINFO_SOURCERANGE_H
#define KERMA_SOURCEINFO_SOURCERANGE_H

#include "kerma/SourceInfo/SourceLoc.h"

namespace kerma {

class SourceRange {
private:
  SourceLoc S;
  SourceLoc E;

public:
  SourceRange();
  SourceRange(SourceLoc start);
  SourceRange(SourceLoc start, SourceLoc end);

  SourceLoc & getStart();
  SourceLoc & getEnd();

  /// A range is valid if its start location is valid
  /// When the end location is invalid, it generally
  /// means until the end of file.
  bool isValid() const;
  bool isInvalid() const;

  bool operator==(const SourceRange& other) const;
  bool operator!=(const SourceRange& other) const;

  friend std::ostream & operator<<(std::ostream &os, const SourceRange &loc);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, SourceRange &loc);
};

} // kerma

#endif // KERMA_SOURCEINFO_SOURCERANGE_H