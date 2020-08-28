#ifndef KERMA_BASE_INDEX_H
#define KERMA_BASE_INDEX_H

#include "llvm/Support/raw_ostream.h"

namespace kerma {

class Dim;

/// Represents a 3-dimensional Index
/// The maximum representable Index is (UINT_MAX - 1, UINT_MAX - 1, UINT_MAX - 1)
/// The index (UINT_MAX,UINT_MAX,UINT_MAX) is used to denote an unknown index
class Index {
public:
  Index(unsigned int z, unsigned int y, unsigned int x);
  Index(unsigned int y, unsigned int x);
  Index(unsigned int x);
  Index();
  Index(const Index &other);
  Index(const Index &&other);
  virtual ~Index()=default;

public:
  //===-------
  // Conversion operators
  //===-------
  virtual operator bool() const;
  //===-------
  // Comparison operators (Legicographic comparison. Order (z,y,x))
  //===-------
  virtual bool operator==(const Index& other) const;
  virtual bool operator!=(const Index& other) const;
  virtual bool operator<(const Index& other) const;
  virtual bool operator<=(const Index& other) const;
  virtual bool operator>(const Index& other) const;
  virtual bool operator>=(const Index& other) const;
  //===-------
  // Inc/Dec operators
  //===-------
  virtual Index& operator++();
  virtual Index& operator++(int);
  virtual Index& operator--();
  virtual Index& operator--(int);
  //===-------
  // Arithmetic operators
  //===-------
  virtual Index operator+(const Index& other) const;
  virtual Index operator-(const Index& other) const;
  virtual Index& operator+=(const Index& other);
  virtual Index& operator-=(const Index& other);

  friend std::ostream& operator<<(std::ostream& os, const Index& idx);
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Index& idx);

public:
  virtual Index& inc(unsigned int x);
  virtual Index& inc(unsigned int y, unsigned int x);
  virtual Index& inc(unsigned int z, unsigned int y, unsigned int x);

  virtual Index& dec(unsigned int x);
  virtual Index& dec(unsigned int y, unsigned int x);
  virtual Index& dec(unsigned int z, unsigned int y, unsigned int x);

  virtual unsigned int col() const;
  virtual unsigned int row() const;
  virtual unsigned int layer() const;

  virtual unsigned long long getLinear(const Dim& dim) const;

  virtual bool isUnknown() const;

public:
  unsigned int x;
  unsigned int y;
  unsigned int z;

  static const Index Zero;
  static const Index Unknown;

  /// Create a linear index out of a multi-dimensional one
  /// \throw std::out_of_range
  static unsigned long long linearize(const Index &idx, const Dim &dim);

  /// Create a 3d index out a linear one
  /// \throw std::out_of_range
  static Index delinearize(unsigned long long idx, const Dim &dim);
};



} // namespace kerma

#endif