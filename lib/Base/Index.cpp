#include "kerma/Base/Index.h"
#include "kerma/Base/Dim.h"
#include <climits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

namespace kerma {

Index::Index() : Index(0) 
{}

Index::Index(unsigned int x) : x(x), y(0), z(0)
{}

Index::Index(unsigned int y, unsigned int x) : x(x), y(y), z(0)
{}

Index::Index(unsigned int z, unsigned int y, unsigned int x)
: x(x), y(y), z(z)
{}

Index::Index(const Index &other) : x(other.x), y(other.y), z(other.z)
{}

Index::Index(const Index &&other) : x(other.x), y(other.y), z(other.z)
{}

//===-------
// Conversion operators
//===-------
Index::operator bool() const {
  return !this->isUnknown();
}

//===-------
// Comparison operators
//===-------
bool Index::operator==(const Index &other) const {
  return this->x == other.x && this->y == other.y && this->z == other.z;
}

bool Index::operator!=(const Index &other) const {
  return !(*this == other);
}

bool Index::operator<(const Index &other) const {
  if ( this->z == other.z) {
    if ( this->y == other.y)
      return this->x < other.x;
    return this->y < other.y;
  }
  return this->z < other.z;
}

bool Index::operator<=(const Index &other) const {
  return (*this == other) || (*this < other);
}

bool Index::operator>(const Index &other) const {
  if ( this->z == other.z) {
    if ( this->y == other.y)
      return this->x > other.x;
    return this->y > other.y;
  }
  return this->z > other.z;
}

bool Index::operator>=(const Index &other) const {
  return (*this == other) || (*this > other);
}

//===-------
// Inc/Dec operators
//===-------
Index& Index::operator++() {
  this->x++;
  return *this;
}

Index& Index::operator++(int) {
  this->x++;
  return *this;
}

Index& Index::operator--() {
  this->x--;
  return *this;
}

Index& Index::operator--(int) {
  this->x--;
  return *this;
}

//===-------
// Arithmetic operators
//===-------
Index Index::operator+(const Index& other) const {
  return Index(this->z + other.z, this->y + other.y, this->x + other.x);
}

Index Index::operator-(const Index& other) const {
  return Index(this->z - other.z, this->y - other.y, this->x - other.x);
}

Index& Index::operator+=(const Index& other) {
  this->x += other.x;
  this->y += other.y;
  this->z += other.z;
  return *this;
}

Index& Index::operator-=(const Index &other) {
  this->x -= other.x;
  this->y -= other.y;
  this->z -= other.z;
  return *this;
}

std::ostream& operator<<(std::ostream& os, const Index& idx) {
  os << "(" << idx.z << "," << idx.y << "," << idx.x << ")";
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index& idx) {
  os << "(" << idx.z << "," << idx.y << "," << idx.x << ")";
  return os;
}

bool Index::isUnknown() const {
  return *this == Index::Unknown;
}

inline Index& Index::inc(unsigned int x) {
  return this->inc(0, 0, x);
}

inline Index& Index::inc(unsigned int y, unsigned int x) {
  return this->inc(0,y,x);
}

inline Index& Index::inc(unsigned int z, unsigned int y, unsigned int x) {
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

inline Index& Index::dec(unsigned int z, unsigned int y, unsigned int x) {
  this->x -= x;
  this->y -= y;
  this->z -= z;
  return *this;
}

unsigned int Index::col() const {
  return this->x;
}

unsigned int Index::row() const {
  return this->y;
}

unsigned int Index::layer() const {
  return this->z;
}

unsigned long long Index::getLinear(const Dim& dim) const {
  return Index::linearize(*this, dim);
}

const Index Index::Zero = Index(0,0,0);
const Index Index::Unknown = Index(UINT_MAX, UINT_MAX, UINT_MAX);

unsigned long long Index::linearize(const Index &idx, const Dim &dim) {
  if ( idx.x >= dim.x || idx.y >= dim.y || idx.z >= dim.z) {
    std::stringstream ssIdx, ssDim;
    ssIdx << idx;
    ssDim << dim;
    throw std::out_of_range(std::string("Invalid index ") + ssIdx.str() 
                          + std::string(" for dim ") + ssDim.str());
  }

  return idx.z * dim.y * dim.x
       + idx.y * dim.x
       + idx.x;
}

Index Index::delinearize(unsigned long long idx, const Dim &dim) {
  if ( idx >= (unsigned long long) dim ) {
    std::stringstream ssIdx, ssDim;
    ssIdx << idx;
    ssDim << dim;
    throw std::out_of_range(std::string("Invalid index ") + std::to_string(idx) 
                          + std::string(" for dim ") + ssDim.str());
  }

  unsigned int z = idx / (dim.y * dim.x);
  idx -= (z * dim.y * dim.x);
  unsigned int y = idx / dim.x;
  unsigned int x = idx % dim.x;
  return Index(z,y,x);
}

} // namespace kerma