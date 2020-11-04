#ifndef KERMA_SOURCEINFO_SOURCEINFO_H
#define KERMA_SOURCEINFO_SOURCEINFO_H

#include "kerma/SourceInfo/SourceRange.h"
#include <string>
#include <unordered_map>

namespace kerma {

class SourceInfo {
  friend class SourceInfoBuilder;

private:
  std::unordered_map<std::string, SourceRange> FunctionRanges;

protected:
  void clear() {
    FunctionRanges.clear();
  }

  void addFunction(const std::string& Name, const SourceRange& Range) {
    FunctionRanges[Name] = Range;
  }

  void addFunctions( const std::unordered_map<std::string, SourceRange> FunctionRangeMap) {
    FunctionRanges.insert(FunctionRangeMap.begin(), FunctionRangeMap.end());
  }
  // void AddStmt(const std::);


public:
  SourceInfo()=default;
  ~SourceInfo()=default;

  SourceInfo& operator=(const SourceInfo& O) {
    FunctionRanges = O.FunctionRanges;
    return *this;
  }

  const std::string& getFunctionOfLine(unsigned int Line) const;
  const SourceRange& getFunctionRange(const std::string& FunctionName) const;

  const std::pair<std::string, SourceRange> getFunctionRangePair(const std::string& FunctionName);
  const std::unordered_map<std::string, SourceRange> getFunctionRanges() const { return FunctionRanges; }
};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCEINFO_H