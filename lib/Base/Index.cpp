#include "kerma/Base/Index.h"
#include <stdexcept>
#include <string>

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

bool Index::operator==(const Index &other) const {
  return this->x == other.x && this->y == other.y && this->z == other.z;
}

inline Index& Index::inc(unsigned int x) {
  return this->inc(0, 0, x);
}

inline Index& Index::inc(unsigned int y, unsigned int x) {
  return this->inc(0,y,x);
}

inline Index& Index::inc(unsigned int x, unsigned int y, unsigned int z) {
  this->x += x;
  this->y += y;
  this->z += z;
  return *this;
}

inline Index& Index::dec(unsigned int x) {
  return this->dec(0,0,x);
}

inline Index& Index::dec(unsigned int y, unsigned int x) {
  return this->dec(0,y,x);
}

inline Index& Index::dec(unsigned int x, unsigned int y, unsigned int z) {
  this->x -= x;
  this->y -= y;
  this->z -= z;
  return *this;
}

const Index Index::Zero = Index(0,0,0);

} // namespace kerma