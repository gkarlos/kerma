#ifndef KERMA_BASE_DIM_H
#define KERMA_BASE_DIM_H

#include "llvm/Support/raw_os_ostream.h"
#include <memory>
#include <ostream>

namespace kerma {

class Index;

/// This class represents dimensionality info
/// It is used to describe the size of 1,2 or 3-dimensional objects
/// Base class for all dim-like classes (e.g CuDim)
class Dim {
public:
  Dim(unsigned int x=1, unsigned int y=1, unsigned int z=1);
  Dim(const Dim &other);
  Dim(const Dim &&other);
  virtual ~Dim()=default;

public:
  //===-------
  // Equality comparison operators
  //===-------
  virtual bool operator==(const Dim &other) const;
  virtual bool operator!=(const Dim &other) const;
  
  //===-------
  // Size comparison operators
  //===-------
  virtual bool operator<(const Dim &other) const;
  virtual bool operator<=(const Dim &other) const;
  virtual bool operator>(const Dim &other) const;
  virtual bool operator>=(const Dim &other) const;
  
  //===-------
  // Conversion operators
  //===-------
  explicit virtual operator bool() const;
  explicit virtual operator unsigned int() const;

  //===-------
  // Other operators
  //===-------
  virtual Dim& operator=(const Dim &other);

  // Index 0 refers to the x-dimension, 1 to the y-dimension and 2 to the z-dimension.
  // @throw std::out_of_range {}
  virtual unsigned int operator[](unsigned int idx) const;
  virtual unsigned int operator()(unsigned int idx) const;

  //
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Dim& dim);

  //
  friend std::ostream& operator<<(std::ostream& os, const Dim& dim);

public:

  // Check if unknown
  bool isUnknown() const;

  // Retrieve the size of this dim
  unsigned long long size() const; 

  // Check if the dim is 1-dimensional, 
  // i.e spans only in the x-dimension
  bool is1D() const; 

  // Check if the dim is 2-dimensional,
  // i.e spans in the y-dimension (x-dimension may still be 1)
  bool is2D() const;

  // Check if the dim is 3-dimensional,
  // i.e spans into the z-dimensions (x and y dimensions may still be 1)
  bool is3D() const;

  // Check if the dim grows at most in 1 dimension
  // The difference compared to is1D() is that y and z 
  // are also considered. Example:
  //
  // Dim dimA(1);   // is1D and isEffective1D
  // Dim dimB(1,2); // is2D and isEffective1D
  // Dim dimC(1,1,2); // is3D and isEffective1D
  bool isEffective1D() const;

  // Check if the dim grows exactly in 2 dimensions
  // The difference compared to issD() is that any 
  // combination is considered. Example:
  //
  // Dim dimA(1,2,1); // is2D and isEffective1D
  // Dim dimB(1,2,2); // is3D and isEffective2D
  // Dim dimC(3,1,3); // is3D and isEffective2D
  bool isEffective2D() const;

  // Check if an index is valid index for this dim
  bool hasIndex(const Index& idx) const;
  bool hasIndex(unsigned long long linearIdx) const;

  // Check if a linear index is valid index for this dim
  // The index is first delinearized based on the dim's
  // values.
  bool hasLinearIndex(unsigned long long idx) const;

  // Get the minimum index for this dim
  // (0,0,0) unless overriden
  virtual Index getMinIndex() const;

  // Get the maximum index for this dim (inclusive).
  // (Z-1,Y-1,X-1) unless overriden
  virtual Index getMaxIndex() const;

  // Get the mimimum linear index for this dim
  // 0 unless overriden
  virtual unsigned long long getMinLinearIndex() const;

  // Get the maximum linear index for this dim
  virtual unsigned long long getMaxLinearIndex() const;
  
public:
  unsigned int x,y,z;

public:
  static Dim of(unsigned int x, unsigned int y, unsigned int z);

  /// Special DIM used to indicated erroneous dim or 
  /// unknown dim
  static const Dim None;
  
  /// The unit dim
  static const Dim Unit;

  ///
  static const Dim& Linear1;
  static const Dim Linear2;
  static const Dim Linear4;
  static const Dim Linear8;
  static const Dim Linear16;
  static const Dim Linear32;
  static const Dim Linear64;
  static const Dim Linear128;
  static const Dim Linear256;
  static const Dim Linear512;
  static const Dim Linear1024;

  static const Dim& Square1;
  static const Dim Square2;
  static const Dim Square4;
  static const Dim Square8;
  static const Dim Square16;
  static const Dim Square32;
  static const Dim Square64;
  static const Dim Square128;
  static const Dim Square256;
  static const Dim Square512;
  static const Dim Square1024;

  static const Dim& Cube1;
  static const Dim Cube2;
  static const Dim Cube4;
  static const Dim Cube8;
  static const Dim Cube16;
  static const Dim Cube32;
  static const Dim Cube64;
  static const Dim Cube128;
  static const Dim Cube256;
  static const Dim Cube512;
  static const Dim Cube1024;
};

} // namespace kerma

#endif