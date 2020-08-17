#ifndef KERMA_BASE_DIM_H
#define KERMA_BASE_DIM_H

#include <memory>

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
  virtual ~Dim();

public:
  
  /// Compare two dims for equality
  /// Two dims are equals when all their components (x,y,z) are equal
  virtual bool operator==(const Dim &other);
  ///
  virtual bool operator!=(const Dim &other);
  /// lexicographic comparison. equivalent to comparing linear indices
  virtual bool operator<(const Dim &other);
  ///
  virtual bool operator<=(const Dim &other) ;
  ///
  virtual bool operator>(const Dim &other);
  ///
  virtual bool operator>=(const Dim &other);
  ///
  virtual operator bool() const;

  /// Index into this Dim. Index 0 refers to the x-dimension,
  /// 1 to the y-dimension and 2 to the z-dimension. On error,
  /// (e.g index out of range) 0 is returned
  virtual unsigned int operator[](unsigned int idx);

public:

  /// Check if unknown
  bool isUnknown();

  /// Retrieve the size of this dim
  unsigned long long size(); 

  /// Check if the dim is 1-dimensional, 
  /// i.e spans only in the x-dimension
  bool is1D(); 

  /// Check if the dim is 2-dimensional,
  /// i.e spans in the y-dimension (x-dimension may still be 1)
  bool is2D();

  /// Check if the dim is 3-dimensional,
  /// i.e spans into the z-dimensions (x and y dimensions may still be 1)
  bool is3D();

  /// Check if the dim grows at most in 1 dimension
  /// The difference compared to is1D() is that y and z 
  /// are also considered. Example:
  ///
  /// Dim dimA(1);   // is1D and isEffective1D
  /// Dim dimB(1,2); // is2D and isEffective1D
  /// Dim dimC(1,1,2); // is3D and isEffective1D
  bool isEffective1D();

  /// Check if the dim grows exactly in 2 dimensions
  /// The difference compared to issD() is that any 
  /// combination is considered. Example:
  ///
  /// Dim dimA(1,2,1); // is2D and isEffective1D
  /// Dim dimB(1,2,2); // is3D and isEffective2D
  /// Dim dimC(3,1,3); // is3D and isEffective2D
  bool isEffective2D();

  /// Check if an index is valid index for this dim
  bool hasIndex(const Index& idx);

  bool hasIndex(unsigned int linearIdx);

  /// Check if a linear index is valid index for this dim
  /// The index is first delinearized based on the dim's
  /// values.
  bool hasLinearIndex(unsigned int idx);

  /// Increase the dimensions of this dim
  virtual Dim& inc(unsigned int x, unsigned int y, unsigned int z);

  /// Decrease the dimensions of this dim
  virtual Dim& dec(unsigned int x, unsigned int y, unsigned int z);

  /// Get the minimum index for this dim
  /// (0,0,0) unless overriden
  virtual Index getMinIndex();

  /// Get the maximum index for this dim
  /// (Z-1,Y-1,X-1) unless overriden
  virtual Index getMaxIndex();

  /// Get the mimimum linear index for this dim
  /// 0 unless overriden
  virtual unsigned long long getMinLinearIndex();

  /// Get the maximum linear index for this dim
  virtual unsigned long long getMaxLinearIndex();
  
public:
  /// size of the x-dimension
  unsigned int x; 
  /// size of the y-dimension
  unsigned int y;
  /// size of the z-dimension
  unsigned int z;

public:
  static Dim of(unsigned int x, unsigned int y, unsigned int z);

  static Dim getLinear256();
  static Dim getLinear512();
  static Dim getLinear1024();
  static Dim getSquare256x256();
  static Dim getSquare512x512();
  static Dim getSquare1024x1024();
  static Dim getRect256x512();
  static Dim getRect256x1024();
  static Dim getRect512x256();
  static Dim getRect512x1024();
  static Dim getRect1024x256();
  static Dim getRect1024x512();

  /// Special DIM uses to indicated erroneous dim or 
  /// unknown dim
  static Dim None;

private:
  Dim createDimNone();
};


} // namespace kerma

#endif