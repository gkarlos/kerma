#ifndef KERMA_BASE_DIM_H
#define KERMA_BASE_DIM_H

#include "kerma/Base/Index.h"

#include <memory>

/// Immutable class representing dimensionality info
/// It is used to describe the size of 1,2 or 3-dimensional objects
/// Base class for all dim-like classes (e.g CuDim)
class Dim {
public:
  Dim(unsigned int x=1, unsigned int y=1, unsigned int z=1);
  Dim(const Dim &other);
  Dim(const Dim &&other);
  virtual ~Dim();

public:
  /// 
  virtual void operator=(const Dim &other);
  
  /// Compare two dims for equality
  /// Two dims are equals when all their components (x,y,z) are equal
  virtual bool operator==(const Dim &other);

  /// Check if this size is strictly less than other size
  virtual bool operator<(const Dim &other);

  /// Create a new dim that is the result or adding the 
  /// size of this and other
  virtual std::unique_ptr<Dim> operator+(const Dim &other);

  /// Create a new dim that is the result of subracting
  /// `other` size from this size. If the operations 
  /// results in negative values in any dimension,
  /// nullptr is returned
  virtual std::unique_ptr<Dim> operator-(const Dim &other);

  /// Index into this Dim. Index 0 refers to the x-dimension,
  /// 1 to the y-dimension and 2 to the z-dimension. On error,
  /// (e.g index out of range) 0 is returned
  virtual unsigned int operator[](unsigned int);

public:
  /// Retrieve the size of this dim
  unsigned int size(); 

  /// Check if the dim is 1-dimensional (y == z == 1)
  bool is1D(); 

  /// Check if the dim is 2-dimensional (y > 1 && z == 1)
  bool is2D();

  /// Check if the dim is 3-dimensional (z > 1)
  bool is3D();

  /// Check if the dim grows exactly in 1 dimension
  /// The difference compared to is1D() is that y and z 
  /// are also considered. Example:
  ///
  /// Dim dimA(1);   // is1D and isEffective1D
  /// Dim dimA(1,2); // is2D and isEffective1D
  bool isEffective1D();

  /// Check if the dim grows exactly in 2 dimensions
  /// The difference compared to issD() is that any 
  /// combination is considered. Example:
  ///
  /// Dim dimA(1,2,1); // is2D and isEffective1D
  /// Dim dimB(1,2,2); // is3D and isEffective2D
  bool isEffective2D();

  /// Check if an index is valid index for this dim
  bool hasIndex(const Index& idx);

  /// Check if a linear index is valid index for this dim
  /// The index is first delinearized based on the dim's
  /// values.
  bool hasLinearIndex(unsigned int idx);

  Index getMaxIndex();

public:
  const unsigned int x; 
  const unsigned int y;
  const unsigned int z;
};

#endif