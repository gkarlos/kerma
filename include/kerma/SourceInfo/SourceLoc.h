#ifndef KERMA_SOURCEINFO_SOURCELOC_H
#define KERMA_SOURCEINFO_SOURCELOC_H

#include <llvm/Support/raw_ostream.h>

#include <ostream>

namespace kerma {

/// Represents a location in the source code, defined
/// by a line and column values.
/// The value 0 is used to denote an unknown value
/// (in line with LLVM/Clang)
/// A location may be uknown, i.e have both line and
/// column being zero (0), or have a known line and
/// an unknown column. The oposite is not possible.
/// Any time the line becomes unknown, the column
/// also becomes unknown.
/// The column may become unknown without affecting
/// the line.
class SourceLoc {
private:
  unsigned int L;
  unsigned int C;

public:
  SourceLoc(unsigned int line=0, unsigned int col=0);
  SourceLoc(const SourceLoc& other);
  SourceLoc(SourceLoc&& other);
  ~SourceLoc()=default;

  unsigned int getCol() const { return C; }
  unsigned int getLine() const { return L; }

  /// Set the line and column of this location
  /// If the line is set to 0 (unknown), then
  /// the column is also changed.
  SourceLoc& set(unsigned int line, unsigned int col);

  /// Set the line of this location
  /// If set to 0, the column is also changed to 0
  SourceLoc& setLine(unsigned int line);
  SourceLoc& setCol(unsigned int col);

  /// Check if this is a valid location
  ///
  /// A location is valid if at least its line number is
  /// known. Example: <br/>
  ///         (1,1) => valid,
  ///         (1,0) => valid,
  ///         (0,0) => invalid
  /// Invalid/Unknown locations are used to represent the
  /// absense of a location. For instance if the End of a
  /// SourceRange is invalid, then the End location is the
  /// end of the file
  bool isValid() const;
  bool isInvalid() const;
  SourceLoc& invalidate();

  /// Check if the location is precise.
  /// A precise location has both line and
  /// column known.
  bool isPrecise() const;

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
  friend llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const SourceLoc &loc);

  /// An invalid location to denote the
  /// absence of source location info
  static const SourceLoc Unknown;
};

} // namespace kerma

namespace std {
  template<> struct hash<kerma::SourceLoc> {
    std::size_t operator()(const kerma::SourceLoc& Loc) const;
  };
}

#endif // KERMA_SOURCEINFO_SOURCELOC_H