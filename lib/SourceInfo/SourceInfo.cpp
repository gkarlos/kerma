#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <algorithm>

namespace kerma {

static std::string FunctionNone = "";

const std::string &SourceInfo::getFunctionOfLine(unsigned int Line) const {
  for (auto &E : KernelRanges)
    if (E.second.getStart().getLine() <= Line &&
        E.second.getEnd().getLine() >= Line)
      return E.first;
  for (auto &E : DeviceFunctionRanges)
    if (E.second.getStart().getLine() <= Line &&
        E.second.getEnd().getLine() >= Line)
      return E.first;
  return FunctionNone;
}

const SourceRange &
SourceInfo::getFunctionRange(const std::string &FunctionName) const {
  auto itk = KernelRanges.find(FunctionName);
  if (itk != KernelRanges.end())
    return itk->second;
  auto itd = DeviceFunctionRanges.find(FunctionName);
  if (itd != DeviceFunctionRanges.end())
    return itd->second;
  return SourceRange::Unknown;
}

const SourceRange &SourceInfo::getRangeForLoc(const llvm::DebugLoc &DL) {
  return getRangeForLoc(SourceLoc(DL));
}

std::vector<SourceRange> SourceInfo::getRangesInRange(const SourceRange &R, bool strict) {
  std::vector<SourceRange> Res;
  if ( R) {
    for ( auto Container : Containers) {
      for ( auto &Range : *Container)
        if ( R.contains(Range))
          Res.push_back(Range);
        else if ( !strict && R.overlaps(Range))
          Res.push_back(Range);
    }
  }
  return Res;
}

const SourceRange &SourceInfo::getRangeForLoc(const SourceLoc &L) {
  // We are assuming that the ranges stored do not overlap
  // so we return the first range matching. This, however,
  // really needs to be tested

  auto fullyContains = [&L](SourceRange &R) { return R.contains(L); };
  auto partiallyContains = [&L](SourceRange &R) { return R.containsLine(L); };

  if (auto it = std::find_if(Stmts.begin(), Stmts.end(), fullyContains);
      it != Stmts.end())
    return *it;
  if (auto it = std::find_if(Exprs.begin(), Exprs.end(), fullyContains);
      it != Exprs.end())
    return *it;
  if (auto it = std::find_if(ForHeaders.begin(), ForHeaders.end(), fullyContains);
      it != ForHeaders.end())
    return *it;
  if (auto it = std::find_if(WhileConditions.begin(), WhileConditions.end(), fullyContains);
      it != WhileConditions.end())
    return *it;
  if (auto it = std::find_if(DoConditions.begin(), DoConditions.end(), fullyContains);
      it != DoConditions.end())
    return *it;
  if (auto it = std::find_if(IfConditions.begin(), IfConditions.end(), fullyContains);
      it != IfConditions.end())
    return *it;
  if (auto it = std::find_if(IfInitializers.begin(), IfInitializers.end(), fullyContains);
      it != IfInitializers.end())
    return *it;
  if (auto it = std::find_if(Stmts.begin(), Stmts.end(), partiallyContains);
      it != Stmts.end())
    return *it;
  if (auto it = std::find_if(Exprs.begin(), Exprs.end(), partiallyContains);
      it != Exprs.end())
    return *it;
  if (auto it = std::find_if(ForHeaders.begin(), ForHeaders.end(), partiallyContains);
      it != ForHeaders.end())
    return *it;
  if (auto it = std::find_if(WhileConditions.begin(), WhileConditions.end(), partiallyContains);
      it != WhileConditions.end())
    return *it;
  if (auto it = std::find_if(DoConditions.begin(), DoConditions.end(), partiallyContains);
      it != DoConditions.end())
    return *it;
  if (auto it = std::find_if(IfConditions.begin(), IfConditions.end(), partiallyContains);
      it != IfConditions.end())
    return *it;
  if (auto it = std::find_if(IfInitializers.begin(), IfInitializers.end(), partiallyContains);
      it != IfInitializers.end())
    return *it;

  return SourceRange::Unknown;
}

} // namespace kerma