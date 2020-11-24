#include "kerma/Base/Loop.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <memory>

using namespace llvm;

namespace kerma {

bool LoopNest::classof(const KermaNode *S) { return S->getKind() == NK_Loop; }

} // namespace kerma