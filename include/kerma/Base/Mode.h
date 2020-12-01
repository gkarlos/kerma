#ifndef KERMA_BASE_MODE_H
#define KERMA_BASE_MODE_H

#include <string>
namespace kerma {

enum Mode : unsigned char {
  GRID='g',
  BLOCK='b',
  WARP='w',
  THREAD='t'
};

std::string ModeStr(Mode M);


} // namespace kerma


#endif // KERMA_BASE_MODE_H