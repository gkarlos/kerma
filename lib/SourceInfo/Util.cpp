#include "kerma/SourceInfo/Util.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>

#include <sstream>

namespace kerma {

void parseClangSrcLocStr(const std::string &LocStr, SourceLoc &Loc) {
  std::string token;
  std::vector<std::string> Values;
  std::stringstream iss(LocStr);

  while (std::getline(iss, token, ':'))
    Values.push_back(token);

  Loc.set(std::stoul(Values[0]), std::stoul(Values[1]));
}

SourceLoc parseClangSrcLocStr(const std::string &LocStr) {
  SourceLoc res;
  parseClangSrcLocStr(LocStr, res);
  return res;
}

SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::SourceRange &R) {
  SourceLoc Begin(SM.getPresumedLineNumber(R.getBegin()),
                  SM.getPresumedColumnNumber(R.getEnd()));
  SourceLoc End(SM.getPresumedLineNumber(R.getBegin()),
                SM.getPresumedColumnNumber(R.getEnd()));
  return SourceRange(Begin, End);
}

SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::Stmt &S) {
  SourceLoc Begin(SM.getPresumedLineNumber(S.getBeginLoc()),
                  SM.getPresumedColumnNumber(S.getBeginLoc()));
  SourceLoc End(SM.getPresumedLineNumber(S.getEndLoc()),
                SM.getPresumedColumnNumber(S.getEndLoc()));
  return SourceRange(Begin, End);
}

SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::Decl &D) {
  SourceLoc Begin(SM.getPresumedLineNumber(D.getBeginLoc()),
                  SM.getPresumedColumnNumber(D.getBeginLoc()));
  SourceLoc End(SM.getPresumedLineNumber(D.getEndLoc()),
                SM.getPresumedColumnNumber(D.getEndLoc()));
  return SourceRange(Begin, End);
}

SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::Expr &E) {
  SourceLoc Begin(SM.getPresumedLineNumber(E.getBeginLoc()),
                  SM.getPresumedColumnNumber(E.getBeginLoc()));
  SourceLoc End(SM.getPresumedLineNumber(E.getEndLoc()),
                SM.getPresumedColumnNumber(E.getEndLoc()));
  return SourceRange(Begin, End);
}

SourceRange readClangSourceRange(const clang::SourceRange &Range,
                                 const clang::SourceManager &SourceManager) {
  std::string BeginLocStr = Range.getBegin().printToString(SourceManager);
  std::string EndLocStr = Range.getEnd().printToString(SourceManager);

  BeginLocStr = BeginLocStr.substr(
      BeginLocStr.find(':') + 1,
      BeginLocStr.find(' ') -
          (BeginLocStr.find(':') ? BeginLocStr.find(':') + 1 : 0));
  EndLocStr =
      EndLocStr.substr(EndLocStr.find(':') + 1,
                       EndLocStr.find(' ') -
                           (EndLocStr.find(':') ? EndLocStr.find(':') + 1 : 0));
  try {
    auto Start = parseClangSrcLocStr(BeginLocStr);
    auto End = parseClangSrcLocStr(EndLocStr);
    return SourceRange(Start, End);
  } catch (...) {
    return SourceRange::Unknown;
  }
}

SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::SourceLocation &B,
                           const clang::SourceLocation &E) {
  SourceLoc Begin(SM.getPresumedLineNumber(B), SM.getPresumedColumnNumber(B));
  SourceLoc End(SM.getPresumedLineNumber(E), SM.getPresumedColumnNumber(E));
  return SourceRange(Begin, End);
}

SourceRange GetForStmtInitRange(const clang::SourceManager &SM,
                                const clang::ForStmt &F) {
  SourceLoc Begin(SM.getPresumedLineNumber(F.getInit()->getBeginLoc()),
                  SM.getPresumedColumnNumber(F.getInit()->getBeginLoc()));
  SourceLoc End(SM.getPresumedLineNumber(F.getInit()->getEndLoc()),
                SM.getPresumedColumnNumber(F.getInit()->getEndLoc()));
  return SourceRange(Begin, End);
}

SourceRange GetForStmtHeaderRange(const clang::SourceManager &SM,
                                  const clang::ForStmt &F) {
  SourceLoc Begin(SM.getPresumedLineNumber(F.getCond()->getBeginLoc()),
                  SM.getPresumedColumnNumber(F.getCond()->getBeginLoc()));
  SourceLoc End(SM.getPresumedLineNumber(F.getRParenLoc()),
                SM.getPresumedColumnNumber(F.getRParenLoc()));
  return SourceRange(Begin, End);
}

} // namespace kerma