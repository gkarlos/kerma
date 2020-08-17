#include "kerma/Base/Index.h"

namespace kerma {

Index::Index() : Index(0) 
{}

Index::Index(unsigned int x) : Index(0, x) 
{}

Index::Index(unsigned int y, unsigned int x) : Index(0, y, x) 
{}

Index::Index(unsigned int z, unsigned int y, unsigned int x)
: z(z), y(y), x(x)
{}

Index::Index(const Index &other) : x(other.x), y(other.y), z(other.z)
{}

Index::Index(const Index &&other) : x(other.x), y(other.y), z(other.z)
{}

const Index Index::Zero = Index(0,0,0);

} // namespace kerma