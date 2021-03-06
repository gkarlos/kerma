#ifndef KERMA_SOURCEINFO_UTIL_H
#define KERMA_SOURCEINFO_UTIL_H

#include "kerma/SourceInfo/SourceLoc.h"
#include "kerma/SourceInfo/SourceRange.h"

#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"

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

SourceRange GetSourceRange(const clang::SourceManager &SM, const clang::SourceRange &R);
SourceRange GetSourceRange(const clang::SourceManager &SM, const clang::Decl &D);
SourceRange GetSourceRange(const clang::SourceManager &SM, const clang::Stmt &S);
SourceRange GetSourceRange(const clang::SourceManager &SM, const clang::Expr &E);
SourceRange GetSourceRange(const clang::SourceManager &SM,
                           const clang::SourceLocation &B,
                           const clang::SourceLocation &E);

SourceRange GetForStmtInitRange(const clang::SourceManager &SM, const clang::ForStmt &F);
SourceRange GetForStmtHeaderRange(const clang::SourceManager &SM, const clang::ForStmt &For);

}

#endif // KERMA_SOURCEINFO_UTIL_H