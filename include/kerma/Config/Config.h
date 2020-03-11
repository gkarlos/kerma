/*!
 * The main config header. Includes all other config headers
 * in this directory.
 */
#ifndef KERMA_CONFIG_CONFIG_H
#define KERMA_CONFIG_CONFIG_H

#include "version.h"
#include "config.h"

#include <cstdlib>

namespace kerma
{
namespace config
{
namespace ver
{
  static constexpr unsigned int stou(const char * str, int val = 0) {
    return (*str ? stou(str + 1, (*str - '0') + val * 10) : val);
  }

  static constexpr float iconcat(const unsigned int a, const unsigned int b) {
    return a + b / 100.0;
  }

  constexpr unsigned int maj = stou(KERMA_MAJOR_VERSION);
  constexpr unsigned int min = stou(KERMA_MINOR_VERSION);
  constexpr float ver = iconcat(maj, min);

} /* NAMESPACE ver    */
} /* NAMESPACE config */
} /* NAMESPACE kerma  */
#endif /* KERMA_CONFIG_CONFIG_H */