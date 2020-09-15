#include "kerma/Support/Parse.h"

#include <sstream>

namespace kerma {

std::vector<std::string>
parseDelimStr(const std::string& s, char delim, std::string(*cb)(const std::string& subStr)) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  while(ss.good()) {
    std::string substr;
    getline(ss, substr, delim);
    res.push_back(cb? cb(substr) : substr);
  }
  return res;
}

}