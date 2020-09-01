#include "kerma/Support/FileUtils.h"

#include <fstream>

namespace kerma {

std::string readTextFile(const char *path) {
  std::ifstream ifs(path);
  if (!ifs.good())
    throw std::runtime_error(std::string("Cannot read file ") + path);
  return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

} // namespace kerma