#include "kerma/Base/Mode.h"

namespace kerma {

std::string ModeStr(Mode M) {
  if (M == THREAD)
    return "thread";
  else if (M == WARP)
    return "warp";
  else if ( M == BLOCK)
    return "block";
  else
    return "grid";
}

}