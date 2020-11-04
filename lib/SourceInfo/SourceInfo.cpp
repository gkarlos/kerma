#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"

namespace kerma {

static std::string FunctionNone = "";

const std::string&
SourceInfo::getFunctionOfLine(unsigned int Line) const {
  for ( auto &E : FunctionRanges)
    if ( E.second.getStart().getLine() <= Line &&
          E.second.getEnd().getLine() >= Line )
        return E.first;
  return FunctionNone;
}

const SourceRange&
SourceInfo::getFunctionRange(const std::string& FunctionName) const {
  auto it = FunctionRanges.find(FunctionName);
  return (it != FunctionRanges.end()) ? it->second : SourceRange::Unknown;
}

} // namespace kerma