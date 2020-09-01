#ifndef KERMA_SUPPORT_FILEUTILS_H
#define KERMA_SUPPORT_FILEUTILS_H

#include <memory>
namespace kerma {

/// Reads a text file and returns a string with 
/// its contents.
///
/// @param path Path to the file
/// @throw std::runtime_error If the file cannot be read
/// @returns std::string
std::string readTextFile(const char *path);

} // namespace kerma

#endif // KERMA_SUPPORT_FILEUTILS_H