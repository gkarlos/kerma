/*! @file include/kerma/Config/Config.h
 *
 * The main config header. It is meant to be THE ONLY config header
 * to include when config info is needed. It includes all other config 
 * headers and provides wrappers to the macros they define.
 */
#ifndef KERMA_CONFIG_CONFIG_H
#define KERMA_CONFIG_CONFIG_H

#include "version.h"
#include "config.h"

namespace kerma
{
namespace config
{
namespace ver
{
  // Version
  static constexpr unsigned int stou(const char * str, int val = 0) {
    return (*str ? stou(str + 1, (*str - '0') + val * 10) : val);
  }

  static constexpr float iconcat(const unsigned int a, const unsigned int b) {
    return a + b / 100.0;
  }

  
  constexpr unsigned int maj = stou(KERMA_MAJOR_VERSION);
  constexpr unsigned int min = stou(KERMA_MINOR_VERSION);
  constexpr float full = iconcat(maj, min);
} /* NAMESPACE ver    */
  
// Tests
constexpr bool testsEnabled = static_cast<bool>(KERMA_TESTS_ENABLED);

// Examples
constexpr bool examplesEnabled = static_cast<bool>(KERMA_EXAMPLES_ENABLED);

} /* NAMESPACE config */
} /* NAMESPACE kerma  */
#endif /* KERMA_CONFIG_CONFIG_H */