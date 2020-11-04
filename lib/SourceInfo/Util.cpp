#include "kerma/SourceInfo/Util.h"
#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"

#include <sstream>

namespace kerma {

void parseClangSrcLocStr( const std::string& LocStr, SourceLoc& Loc) {
  std::string token;
  std::vector<std::string> Values;
  std::stringstream iss(LocStr);

  while ( std::getline(iss, token, ':'))
    Values.push_back(token);

  Loc.set( std::stoul(Values[0]), std::stoul(Values[1]));
}

SourceLoc parseClangSrcLocStr( const std::string& LocStr) {
  SourceLoc res;
  parseClangSrcLocStr(LocStr, res);
  return res;
}

SourceRange readClangSourceRange(const clang::SourceRange &Range, clang::SourceManager& SourceManager) {
  std::string BeginLocStr = Range.getBegin().printToString(SourceManager);
  std::string EndLocStr = Range.getEnd().printToString(SourceManager);

  BeginLocStr = BeginLocStr.substr(BeginLocStr.find(':') + 1,
                                    BeginLocStr.find(' ') - (BeginLocStr.find(':')? BeginLocStr.find(':') + 1 : 0));
  EndLocStr = EndLocStr.substr(EndLocStr.find(':') + 1,
                                EndLocStr.find(' ') - (EndLocStr.find(':')? EndLocStr.find(':') + 1 : 0));
  try {
    auto Start = parseClangSrcLocStr(BeginLocStr);
    auto End = parseClangSrcLocStr(EndLocStr);
    return SourceRange(Start, End);
  } catch (...) {
    return SourceRange::Unknown;
  }
}

} // namespace kerma