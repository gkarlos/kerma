#ifndef KERMA_SUPPORT_CXXEXTRAS_H
#define KERMA_SUPPORT_CXXEXTRAS_H

#include <algorithm>
#include <vector>
#include <string>

namespace kerma {

/// Check if a vector t contains an element t
template<typename T> bool inVector(const T& t, const std::vector<std::string>& v) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

}


//} // namespace kerma

#endif // KERMA_SUPPORT_CXXEXTRAS_H