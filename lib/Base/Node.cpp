#include "kerma/Base/Node.h"

#include <mutex>

namespace kerma {

static std::mutex mtx;

unsigned KermaNode::genID() {
  static volatile unsigned int IDs = 0;
  unsigned int id;
  mtx.lock();
  id = IDs++;
  mtx.unlock();
  return id;
}

// llvm::raw_ostream &operator<<(llvm::raw_ostream &O, const KermaNode &KN) {
//   KN.print(O);
//   return O;
// }

} // namespace kerma