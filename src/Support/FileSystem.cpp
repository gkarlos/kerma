//
// Created by gkarlos on 21-01-20.
//

#include <llvm/Support/FileSystem.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <cstdlib>

namespace kerma {

std::string get_cwd()
{
  char temp[ PATH_MAX ];

  if ( getcwd(temp, PATH_MAX) != 0)
    return std::string ( temp );

  int error = errno;

  switch ( error ) {
    case EACCES:
      throw std::runtime_error("Access denied");

    case ENOMEM:
      // Can this happen?
      throw std::runtime_error("Insufficient storage");

    default: {
      std::ostringstream str;
      str << "Unrecognised error" << error;
      throw std::runtime_error(str.str());
    }
  }
}

std::string
get_realpath(const std::string& relpath)
{
  char real[ PATH_MAX];
  char *res = realpath(relpath.c_str(), real);
  if ( !res)
    throw std::runtime_error("realpath() error");

  return std::string(real);
}

} /// NAMESPACE kerma
