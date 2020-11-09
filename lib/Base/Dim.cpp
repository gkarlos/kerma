#include "kerma/Base/Dim.h"
#include "kerma/Base/Index.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kerma {

Dim::Dim() : Dim(1) {}
Dim::Dim(unsigned x) : Dim((bool) x, x) {}
Dim::Dim(unsigned y, unsigned x) : Dim(y && x, y, x) {}
Dim::Dim(unsigned z, unsigned y, unsigned x) : x(x), y(y), z(z) {
  if ( !x || !y || !z) {
    this->x = 0;
    this->y = 0;
    this->z = 0;
  }
}

Dim::Dim(const Dim &other) : x(other.x), y(other.y), z(other.z) {}

Dim::Dim(const Dim &&other) : x(other.x), y(other.y), z(other.z) {}

//===-------
// Equality comparison operators
//===-------

bool Dim::operator==(const Dim &other) const {
  return this->x == other.x && this->y == other.y && this->z == other.z;
}

bool Dim::operator!=(const Dim &other) const { return !(*this == other); }

//===-------
// Conversion operators
//===-------
Dim::operator unsigned long long() const { return this->size(); }

//===-------
// Other operators
//===-------

Dim &Dim::operator=(const Dim &other) {
  if (this != &other) {
    x = other.x;
    y = other.y;
    z = other.z;
  }
  return *this;
}

unsigned int Dim::operator[](unsigned int idx) const {
  switch (idx) {
  case 0:
    return this->x;
  case 1:
    return this->y;
  case 2:
    return this->z;
  default:
    throw std::out_of_range(std::string("Index ") + std::to_string(idx) +
                            " is out of range");
  }
}

unsigned int Dim::operator()(unsigned int idx) const {
  switch (idx) {
  case 0:
    return this->x;
  case 1:
    return this->y;
  case 2:
    return this->z;
  default:
    throw std::out_of_range(std::string("Index ") + std::to_string(idx) +
                            " is out of range");
  }
}

std::ostream &operator<<(std::ostream &os, const Dim &dim) {
  if ( dim.isUnknown())
    os << "<>";
  else if (dim.is1D())
    os << "<" << dim.x << ">";
  else if (dim.is2D())
    os << "<" << dim.y << "," << dim.x << ">";
  else
    os << "<" << dim.z << "," << dim.y << "," << dim.x << ">";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Dim &dim) {
  if ( dim.isUnknown())
    os << "<>";
  else if (dim.is1D())
    os << "<" << dim.x << ">";
  else if (dim.is2D())
    os << "<" << dim.y << "," << dim.x << ">";
  else
    os << "<" << dim.z << "," << dim.y << "," << dim.x << ">";
  return os;
}

std::string Dim::toString() const {
  std::stringstream ss;
  if ( isUnknown())
    ss << "<>";
  else if (is1D())
    ss << "<" << x << ">";
  else if (is2D())
    ss << "<" << y << "," << x << ">";
  else
    ss << "<" << z << "," << y << "," << x << ">";
  return ss.str();
}

bool Dim::isUnknown() const { return !*this; }

unsigned long long Dim::size() const { return this->x * this->y * this->z; }

bool Dim::is1D() const { return this->y == 1 && this->z == 1; }

bool Dim::is2D() const { return this->y > 1 && this->z == 1; }

bool Dim::is3D() const { return this->z > 1; }

bool Dim::isEffective1D() const {
  return this->is1D() || (this->x == 1 && this->y == 1 && this->z > 1) ||
         (this->x == 1 && this->y > 1 && this->z == 1) ||
         (this->x > 1 && this->y == 1 && this->z == 1);
}

bool Dim::isEffective2D() const {
  return (this->x > 1 && this->y > 1 && this->z == 1) ||
         (this->x > 1 && this->z > 1 && this->y == 1) ||
         (this->y > 1 && this->z > 1 && this->x == 1);
}

bool Dim::hasIndex(const Index &idx) const {
  return idx.x < this->x && idx.y < this->y && idx.z < this->z;
}

bool Dim::hasIndex(unsigned long long linearIdx) const {
  return this->hasLinearIndex(linearIdx);
}

bool Dim::hasLinearIndex(unsigned long long idx) const {
  return idx < this->size();
}

Index Dim::getMinIndex() const {
  static Index MinIndex = Index(0, 0, 0);
  return MinIndex;
}

Index Dim::getMaxIndex() const {
  if (this->isUnknown())
    return Index(0, 0, 0);
  return Index(this->z - 1, this->y - 1, this->x - 1);
}

unsigned long long Dim::getMinLinearIndex() const { return 0; }

unsigned long long Dim::getMaxLinearIndex() const { return this->size() - 1; }

unsigned int Dim::cols() const { return this->x; }

unsigned int Dim::rows() const { return this->y; }

unsigned int Dim::layers() const { return this->z; }

unsigned int Dim::width() const { return this->x; }

unsigned int Dim::height() const { return this->y; }

unsigned int Dim::depth() const { return this->z; }


const Dim Dim::None(0);
const Dim Dim::Unit = Dim();

const Dim Dim::Linear2(2);
const Dim Dim::Linear4(4);
const Dim Dim::Linear8(8);
const Dim Dim::Linear16(16);
const Dim Dim::Linear32(32);
const Dim Dim::Linear64(64);
const Dim Dim::Linear128(128);
const Dim Dim::Linear256(256);
const Dim Dim::Linear512(512);
const Dim Dim::Linear1024(1024);

const Dim Dim::Square2(2, 2);
const Dim Dim::Square4(4, 4);
const Dim Dim::Square8(8, 8);
const Dim Dim::Square16(16, 16);
const Dim Dim::Square32(32, 32);
const Dim Dim::Square64(64, 64);
const Dim Dim::Square128(128, 128);
const Dim Dim::Square256(256, 256);
const Dim Dim::Square512(512, 512);
const Dim Dim::Square1024(1024, 1024);

const Dim Dim::Cube2(2, 2, 2);
const Dim Dim::Cube4(4, 4, 4);
const Dim Dim::Cube8(8, 8, 8);
const Dim Dim::Cube16(16, 16, 16);
const Dim Dim::Cube32(32, 32, 32);
const Dim Dim::Cube64(64, 64, 64);
const Dim Dim::Cube128(128, 128, 128);
const Dim Dim::Cube256(256, 256, 256);
const Dim Dim::Cube512(512, 512, 512);
const Dim Dim::Cube1024(1024, 1024, 1024);
} // namespace kerma