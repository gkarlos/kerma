#ifndef KERMA_SOURCEINFO_SOURCERANGE_H
#define KERMA_SOURCEINFO_SOURCERANGE_H

#include "kerma/SourceInfo/SourceLoc.h"

namespace kerma {

class SourceRange {
private:
  SourceLoc S;
  SourceLoc E;

public:
  SourceRange()=default;
  SourceRange(SourceLoc start);
  SourceRange(SourceLoc start, SourceLoc end);

  void setStart( const SourceLoc& Start) { S = Start; }
  void setEnd( const SourceLoc& End) { E = End; }

  const SourceLoc & getStart() const { return S; }
  const SourceLoc & getEnd() const { return E; }

  /// A range is valid if its start location is valid
  /// When the end location is invalid, it generally
  /// means until the end of file.
  bool isValid() const;
  bool isInvalid() const;

  /// Check if the range fully contains
  /// another range
  bool contains(const SourceRange &Range) const;

  /// Check if the range contains a location
  bool contains(const SourceLoc &Loc) const;

  /// Check if the range partially contains a loc
  /// That is the loc's line is within the range
  /// but the column may not be
  bool containsLine(const SourceLoc &Loc) const;

  /// Check if the  range overlaps with another
  /// range. That is, whether either range
  /// fully or partially contains the other
  bool overlaps(const SourceRange &Range) const;

  /// Returns true if the end of the range is
  /// strictly greater than the start
  bool isEmpty() const;

  bool operator==(const SourceRange& other) const;
  bool operator!=(const SourceRange& other) const;
  operator bool() const { return isValid(); }

  friend std::ostream & operator<<(std::ostream &os, const SourceRange &loc);
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const SourceRange &loc);

  static const SourceRange Unknown;
};

} // kerma

namespace std {
  template<> struct hash<kerma::SourceRange> {
    std::size_t operator()(const kerma::SourceRange& Range) const;
  };
}

#endif // KERMA_SOURCEINFO_SOURCERANGE_H