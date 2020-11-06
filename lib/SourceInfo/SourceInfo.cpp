#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"

namespace kerma {

static std::string FunctionNone = "";

const std::string&
SourceInfo::getFunctionOfLine(unsigned int Line) const {
  for ( auto &E : KernelRanges)
    if ( E.second.getStart().getLine() <= Line &&
         E.second.getEnd().getLine() >= Line )
        return E.first;
  for ( auto &E : DeviceFunctionRanges)
    if ( E.second.getStart().getLine() <= Line &&
         E.second.getEnd().getLine() >= Line )
        return E.first;
  return FunctionNone;
}

const SourceRange&
SourceInfo::getFunctionRange(const std::string& FunctionName) const {
  auto itk = KernelRanges.find(FunctionName);
  if (itk != KernelRanges.end())
    return itk->second;
  auto itd = DeviceFunctionRanges.find(FunctionName);
  if (itd != DeviceFunctionRanges.end())
    return itd->second;
  return SourceRange::Unknown;
}

} // namespace kerma