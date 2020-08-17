#include "kerma/Base/Dim.h"
#include "kerma/Base/Index.h"

#include <algorithm> 
#include <memory>
#include <stdexcept>

using namespace kerma;

Dim::Dim(unsigned int x, unsigned int y, unsigned int z)
: x(x), y(y), z(z)
{}

Dim::Dim(const Dim &other) 
: x(other.x), y(other.y), z(other.z)
{}

Dim::Dim(const Dim &&other) 
: x(other.x), y(other.y), z(other.z)
{}

bool Dim::operator==(const Dim &other) {
  return this->x == other.x && this->y == other.y && this->z == other.z;
}

bool Dim::operator!=(const Dim &other) {
  return !(*this == other);
}

bool Dim::operator<(const Dim &other) {
  if ( this->x == other.x)
    if ( this->y == other.y )
      return this->z < other.z;
    return this->y < other.y;
  return this->x < other.x;
}

bool Dim::operator<=(const Dim &other) {
  return (*this == other) || (*this < other);
}

bool Dim::operator>(const Dim &other) {
  return !(*this <= other);
}

bool Dim::operator>=(const Dim &other) {
  return (*this == other) || !(*this < other);
}

Dim::operator bool() const {
  return this->x > 0 && this->y > 0 && this->z > 0;
}

unsigned int Dim::operator[](unsigned int idx) {
  switch ( idx) {
    case 0: return this->x;
    case 1: return this->y;
    case 2: return this->z;
    default:
      return 0;
  }
}

bool Dim::isUnknown() {
  return !*this;
}

unsigned long long Dim::size() {
  return this->x * this->y *this->z;
}

bool Dim::is1D() const {
  return this->y == 1 && this->z == 1;
}

bool Dim::is2D() const {
  return this->y > 1 && this->z  == 1;
}

bool Dim::is3D() const {
  return this->z > 1;
}

bool Dim::isEffective1D() const {
  return this->is1D()
    ||  (this->x == 1 && this->y == 1 && this->z  > 1)
    ||  (this->x == 1 && this->y  > 1 && this->z == 1)
    ||  (this->x  > 1 && this->y == 1 && this->z == 1);
}

bool Dim::isEffective2D() const {
  return (this->x > 1 && this->y > 1 && this->z == 1)
      || (this->x > 1 && this->z > 1 && this->y == 1)
      || (this->y > 1 && this->z > 1 && this->x == 1);
}

bool Dim::hasIndex(const Index& idx) {
  return idx.x < this->x && idx.y < this->y && idx.z < this->z;
}

bool Dim::hasIndex(unsigned int linearIdx) {
  return this->hasLinearIndex(linearIdx);
}

bool Dim::hasLinearIndex(unsigned int idx) {
  return idx < this->size();
}

Dim Dim::inc(unsigned int x, unsigned int y, unsigned int z) {
  return Dim(this->x + x, this->y + y, this->z + z);
}

Dim Dim::dec(unsigned int x, unsigned int y, unsigned int z) {
  unsigned int dx = this->x - x;
  unsigned int dy = this->y - y;
  unsigned int dz = this->z - z;
  if ( dx < 1 || /*overflow*/ dx > x || dy < 1 || /*overflow*/ dy > x ||
       dz < 1 || /*overflow*/ dz > x)
    return Dim(0,0,0);
  return Dim(dx, dy, dz);
}

Index Dim::getMinIndex() {
  static Index MinIndex = Index(0, 0, 0);
  return MinIndex;
}

Index Dim::getMaxIndex() {
  if ( this->isUnknown())
    return Index::Zero;
  return Index(this->x - 1, this->y - 1, this->z - 1);
}

unsigned long long Dim::getMinLinearIndex() {
  return 0;
}

unsigned long long Dim::getMaxLinearIndex() {
  return this->size() - 1;
}

Dim Dim::of(unsigned int x, unsigned int y, unsigned int z) {
  return Dim(x, y, z);
}

const Dim Dim::None = Dim(0,0,0);
const Dim Dim::Unit = Dim(1,1,1);
const Dim Dim::Linear256 = Dim(256,1,1);