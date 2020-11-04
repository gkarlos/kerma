#ifndef KERMA_SOURCEINFO_UTIL_H
#define KERMA_SOURCEINFO_UTIL_H

#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"

#include "clang/Basic/SourceLocation.h"

namespace kerma {

/// Parse a Clang SourceLocation string into a SourceLoc object
void parseClangSrcLocStr( const std::string& LocStr, SourceLoc& Res);
SourceLoc parseClangSrcLocStr( const std::string& LocStr);

// Turn a clang::SourceRange to a kerma::SourceRange
//
// FIXME: A range string has the form:
//        <filename>:<line>:<col>[ <other>]
//                  ^             ^space
// At the moment we extract the substr between the first ':' and the
// first space. If no space exists, then until the end of the string.
// This may fail on malformed strings so a more robust way is needed.
SourceRange readClangSourceRange( const clang::SourceRange &Range, clang::SourceManager& SourceManager);
// void readClangSrcRange( const clang::SourceRange &Range, SourceLoc& Res);

}

#endif // KERMA_SOURCEINFO_UTIL_H