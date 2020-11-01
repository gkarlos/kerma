#ifndef KERMA_TOOLS_KERMAD_UTIL_H
#define KERMA_TOOLS_KERMAD_UTIL_H

#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace kerma {
namespace kermad {

std::string getTimestamp() {
  std::ostringstream oss;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  oss << std::put_time(&tm, "%d%m%Y-%H%M%S");
  return oss.str();
}

}
}


#endif