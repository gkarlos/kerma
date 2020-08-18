#include "kerma/Base/Dim.h"
#include "kerma/Base/Index.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm> 
#include <memory>
#include <ostream>
#include <stdexcept>
#include <iostream>
#include <string>

namespace kerma {

Dim::Dim(unsigned int x, unsigned int y, unsigned int z)
: x(x), y(y), z(z)
{}

Dim::Dim(const Dim &other) 
: x(other.x), y(other.y), z(other.z)
{}

Dim::Dim(const Dim &&other) 
: x(other.x), y(other.y), z(other.z)
{}

Dim& Dim::operator=(const Dim &other) {
  if ( this != &other) {
    x = other.x;
    y = other.y;
    z = other.z;
  }
  return *this;
}

//===-------
// Equality comparison operators
//===-------

bool Dim::operator==(const Dim &other) const {
  return this->x == other.x && this->y == other.y && this->z == other.z;
}

bool Dim::operator!=(const Dim &other) const {
  return !(*this == other);
}

//===-------
// Size comparison operators
//===-------
bool Dim::operator<(const Dim &other) const {
  return this->size() < other.size();
}

bool Dim::operator<=(const Dim &other) const {
  return (*this == other) || (*this < other);
}

bool Dim::operator>(const Dim &other) const {
  return !(*this <= other);
}

bool Dim::operator>=(const Dim &other) const {
  return (*this == other) || !(*this < other);
}

//===-------
// Conversion operators
//===-------
Dim::operator unsigned int() const {
  return this->size();
}

Dim::operator bool() const {
  return this->x > 0 && this->y > 0 && this->z > 0;
}

unsigned int Dim::operator[](unsigned int idx) const {
  switch ( idx) {
    case 0: return this->x;
    case 1: return this->y;
    case 2: return this->z;
    default:
      throw std::out_of_range(std::string("Index ") + std::to_string(idx) + " is out of range");
  }
}

std::ostream& operator<<(std::ostream& os, const Dim& dim) {
  if ( dim.is1D())
    os << "<" << dim.x << ">";
  else if ( dim.is2D())
    os << "<" << dim.x << "," << dim.y << ">";
  else
    os << "<" << dim.x << "," << dim.y << "," << dim.z << ">";
  return os;
}

llvm::raw_os_ostream& operator<<(llvm::raw_os_ostream& os, const Dim& dim) {
  if ( dim.is1D())
    os << "<" << dim.x << ">";
  else if ( dim.is2D())
    os << "<" << dim.x << "," << dim.y << ">";
  else
    os << "<" << dim.x << "," << dim.y << "," << dim.z << ">";
  return os;
}

bool Dim::isUnknown() const {
  return !*this;
}

unsigned long long Dim::size() const {
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

bool Dim::hasIndex(unsigned int linearIdx) const {
  return this->hasLinearIndex(linearIdx);
}

bool Dim::hasLinearIndex(unsigned int idx) const {
  return idx < this->size();
}

Index Dim::getMinIndex() {
  static Index MinIndex = Index(0, 0, 0);
  return MinIndex;
}

Index Dim::getMaxIndex() {
  if ( this->isUnknown())
    return Index(0,0,0);
  return Index(this->x - 1, this->y - 1, this->z - 1);
}

unsigned long long Dim::getMinLinearIndex() const {
  return 0;
}

unsigned long long Dim::getMaxLinearIndex() const {
  return this->size() - 1;
}

Dim Dim::of(unsigned int x, unsigned int y, unsigned int z) {
  return Dim(x, y, z);
}

const Dim Dim::None(0,0,0);
const Dim Dim::Unit = Dim();

const Dim Dim::Linear1 = Dim::Unit;
const Dim Dim::Linear2 = Dim(2);
const Dim Dim::Linear4 = Dim(4);
const Dim Dim::Linear8 = Dim(8);
const Dim Dim::Linear16 = Dim(16);
const Dim Dim::Linear32 = Dim(32);
const Dim Dim::Linear64 = Dim(64);
const Dim Dim::Linear128 = Dim(128);
const Dim Dim::Linear256 = Dim(256);
const Dim Dim::Linear512 = Dim(512);
const Dim Dim::Linear1024 = Dim(1024);

const Dim Dim::Square1 = Dim::Unit;
const Dim Dim::Square2 = Dim(2,2);
const Dim Dim::Square4 = Dim(4,4);
const Dim Dim::Square8 = Dim(8,8);
const Dim Dim::Square16 = Dim(16,16);
const Dim Dim::Square32 = Dim(32,32);
const Dim Dim::Square64 = Dim(64,64);
const Dim Dim::Square128 = Dim(128,128);
const Dim Dim::Square256 = Dim(256,256);
const Dim Dim::Square512 = Dim(512,512);
const Dim Dim::Square1024 = Dim(1024,1024);

const Dim& Dim::Cube1 = Dim::Unit;
const Dim Dim::Cube2(2,2,2);
const Dim Dim::Cube4(4,4,4);
const Dim Dim::Cube8(8,8,8);

} // namespace kerma