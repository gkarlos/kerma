#ifndef KERMA_SUPPORT_VERSION_H
#define KERMA_SUPPORT_VERSION_H

#include "kerma/Support/Version.inc.in"

#include <string>

namespace kerma {

/// Retrieve a string representing the complete kerma version
/// The version is of the form major.minor.patch
std::string getVersion();

/// Retrieve a string representing the major kerma version
std::string getVersionMajor();

/// Retrieve a string representing the minor kerma version
std::string getVersionMinor();

/// Retrieve a string representing the kerma patch
std::string getVersionPatch();

/// Retrieve an LLVM version string of the form major.minor.patch
std::string getLLVMVersion();

}

#endif