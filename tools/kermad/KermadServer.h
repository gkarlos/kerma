#ifndef KERMA_TOOLS_KERMAD_KERMADSERVER_H
#define KERMA_TOOLS_KERMAD_KERMADSERVER_H

#include "KermadOptions.h"

namespace kerma {
namespace kermad {

class KermadServer {

public:
  KermadServer(KermadOptions &ops);
  void start();
  void stop();
};

} // end namespace kermad
} // end namespace kerma

#endif