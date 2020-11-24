#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <algorithm>

namespace kerma {

using namespace llvm;
using namespace std;

static std::string FunctionNone = "";

std::vector<SourceRange> SourceInfo::getIfConditionsInRange(SourceRange Range) {
  std::vector<SourceRange> Res;
  if (Range)
    for (auto &IfStmt : IfStmts)
      if (Range.overlaps(std::get<0>(IfStmt)))
        Res.push_back(std::get<0>(IfStmt));
  return Res;
}

std::vector<std::tuple<SourceRange, SourceRange, SourceRange>>
SourceInfo::getIfRangesInRange(SourceRange R) {
  std::vector<std::tuple<SourceRange, SourceRange, SourceRange>> Res;
  if (R) {
    for (auto &T : IfStmts) {
      if (R.overlaps(get<0>(T))) {
        if (get<1>(T) && R.overlaps(get<1>(T))) {
          if (get<2>(T)) { // has an else
            if (R.overlaps(get<2>(T)))
              Res.push_back(T);
          } else {
            Res.push_back(T);
          }
        }
      }
    }
  }
  return Res;
}

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
  for (auto &E : KernelRanges)
    if (E.first == FunctionName)
      return E.second;
  for (auto &E : DeviceFunctionRanges)
    if (E.first == FunctionName)
      return E.second;
  return SourceRange::Unknown;
}

const SourceRange &SourceInfo::getRangeForLoc(const llvm::DebugLoc &DL) {
  return getRangeForLoc(SourceLoc(DL));
}

std::vector<SourceRange> SourceInfo::getRangesInRange(const SourceRange &R,
                                                      bool strict) {
  std::vector<SourceRange> Res;
  if (R) {
    for (auto Container : Containers) {
      for (auto &Range : *Container)
        if (R.contains(Range))
          Res.push_back(Range);
        else if (!strict && R.overlaps(Range))
          Res.push_back(Range);
    }
  }
  return Res;
}

const SourceRange &SourceInfo::getRangeForLoc(const SourceLoc &L) {
  auto fullyContains = [&L](SourceRange &R) { return R.contains(L); };
  auto partiallyContains = [&L](SourceRange &R) { return R.containsLine(L); };

  if (auto it = std::find_if(Stmts.begin(), Stmts.end(), fullyContains);
      it != Stmts.end())
    return *it;
  if (auto it = std::find_if(Exprs.begin(), Exprs.end(), fullyContains);
      it != Exprs.end())
    return *it;
  if (auto it = std::find_if(ForInits.begin(), ForInits.end(), fullyContains);
      it != ForInits.end())
    return *it;
  if (auto it =
          std::find_if(ForHeaders.begin(), ForHeaders.end(), fullyContains);
      it != ForHeaders.end())
    return *it;
  if (auto it = std::find_if(WhileConditions.begin(), WhileConditions.end(),
                             fullyContains);
      it != WhileConditions.end())
    return *it;
  if (auto it =
          std::find_if(DoConditions.begin(), DoConditions.end(), fullyContains);
      it != DoConditions.end())
    return *it;

  for (auto &If : IfStmts) {
    auto &CondRange = std::get<0>(If);
    if (CondRange.contains(L))
      return CondRange;
  }

  if (auto it = std::find_if(IfInitializers.begin(), IfInitializers.end(),
                             fullyContains);
      it != IfInitializers.end())
    return *it;
  if (auto it = std::find_if(Stmts.begin(), Stmts.end(), partiallyContains);
      it != Stmts.end())
    return *it;
  if (auto it = std::find_if(Exprs.begin(), Exprs.end(), partiallyContains);
      it != Exprs.end())
    return *it;
  if (auto it = std::find_if(ForInits.begin(), ForInits.end(), fullyContains);
      it != ForInits.end())
    return *it;
  if (auto it =
          std::find_if(ForHeaders.begin(), ForHeaders.end(), partiallyContains);
      it != ForHeaders.end())
    return *it;
  if (auto it = std::find_if(WhileConditions.begin(), WhileConditions.end(),
                             partiallyContains);
      it != WhileConditions.end())
    return *it;
  if (auto it = std::find_if(DoConditions.begin(), DoConditions.end(),
                             partiallyContains);
      it != DoConditions.end())
    return *it;

  for (auto &If : IfStmts) {
    auto &CondRange = std::get<0>(If);
    if (CondRange.containsLine(L))
      return CondRange;
  }

  if (auto it = std::find_if(IfInitializers.begin(), IfInitializers.end(),
                             partiallyContains);
      it != IfInitializers.end())
    return *it;

  return SourceRange::Unknown;
}

const SourceRange &SourceInfo::getForInitRangeForLoc(const SourceLoc &L) {
  auto fullyContains = [&L](SourceRange &R) { return R.contains(L); };
  if (auto it = std::find_if(ForInits.begin(), ForInits.end(), fullyContains);
      it != ForInits.end())
    return *it;
  return SourceRange::Unknown;
}

const std::tuple<SourceRange, SourceRange, SourceRange> *
SourceInfo::getRangeForBranch(llvm::BranchInst *BI) {
  if (BI && BI->getDebugLoc()) {
    SourceLoc Loc(BI->getDebugLoc());
    for (auto &IfStmt : IfStmts)
      if (std::get<0>(IfStmt).contains(Loc))
        return &IfStmt;
  }
  return nullptr;
}

llvm::raw_ostream &SourceInfo::print(
    llvm::raw_ostream &OS,
    const std::tuple<SourceRange, SourceRange, SourceRange> &IfRanges) {
  auto &Cond = std::get<0>(IfRanges);
  auto &Then = std::get<1>(IfRanges);
  auto &Else = std::get<2>(IfRanges);

  if (Cond)
    OS << '(' << Cond << ')';
  else
    OS << '-';

  OS << ',';

  if (Then)
    OS << '(' << Then << ')';
  else
    OS << '-';

  if (Else)
    OS << '(' << Else << ')';
  else
    OS << '-';
  return OS;
}

} // namespace kerma