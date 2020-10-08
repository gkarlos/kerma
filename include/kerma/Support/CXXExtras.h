#ifndef KERMA_SUPPORT_CXXEXTRAS_H
#define KERMA_SUPPORT_CXXEXTRAS_H

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <iostream>

namespace kerma {

/// Check if a vector t contains an element t
template<typename T> bool inVector(const T& t, const std::vector<std::string>& v) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

/// Check if a value exists in the map values
template<typename K, typename V>
bool valueInMap(const V& Value, const std::map<K,V>& Map) {
  auto it = Map.begin();
  while ( it != Map.end() ) {
    if ( it->second == Value)
      return true;
  }
  return false;
}

/// Look for a value in a map and collect all keys with
/// that value in a vector. Returns true is there is at
/// least one key with that value found. False otherwise
template<typename K, typename V>
bool findByValue(const V& Value, const std::map<K,V>& Map, std::vector<K>& Keys) {
  auto it = Map.begin();
  bool res = false;
  while ( it != Map.end()) {
    if ( it->second == Value) {
      Keys.push_back(it->first);
      res = true;
    }
  }
  return res;
}

}


//} // namespace kerma

#endif // KERMA_SUPPORT_CXXEXTRAS_H