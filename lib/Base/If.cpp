#include "kerma/Base/If.h"
#include "kerma/Base/Node.h"

namespace kerma {

If::If(const SourceRange &R, KermaNode *Parent)
    : KermaNode(NK_If, R, Parent), DataDep(false) {}

bool If::classof(const KermaNode *S) { return S->getKind() == NK_If; }

} // namespace kerma