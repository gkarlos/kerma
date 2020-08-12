#ifndef KERMA_SUPPORT_IO_H
#define KERMA_SUPPORT_IO_H

#include <string>
#include <system_error>

namespace kerma {

enum IOErr {
  IO_SUCCESS = 0,
  IO_NOT_FOUND,
  IO_IS_DIR,
  IO_INTERNAL_RD_ERR,
  IO_INTERNAL_WR_ERR
};


/*
 * Returns a String describing an IOErr.
 * The empty string is returned on invalid input or error
 */
std::string IOErrToString(IOErr &err);
std::string IOErrToString(IOErr &&err);

/*
 * Read a file and return an std::string with its contents. 
 * Throws runtime exception on error.
 */
std::string readFile(const std::string &path);

/*
 * Read a file and return an std::string with its contents.
 * Does not throw on error. Instead is sets status to a non-zero value
 */
std::string readFile(const std::string &path, std::error_code& status);

/*
 * Write an std::string to a file
 * Throws runtime exception on error.
 */
void writeFile(const std::string &path, const std::string &contents);

/*
 * Write an std::string to file
 * Does not throw on error. Instead it sets status to a non-zero value
 */
void writeFile(const std::string &path, const std::string &contents, std::error_code &status);

}

#endif