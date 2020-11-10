#ifndef KERMA_BASE_DIM_H
#define KERMA_BASE_DIM_H

#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <ostream>

namespace kerma {

class Index;

/// This class represents dimensionality info
/// It is used to describe sizes of objects,
/// in 1, 2 or 3 dimensions.
/// Semantics are similar to CUDA dim3
class Dim {
public:
  /// Default constructor. Creates unit dim (0,0,1)
  Dim();
  /// If x is zero the Dim is unknown
  Dim(unsigned x);
  /// If any of y,z is zero then all become
  /// zero and the Dim is unknown
  Dim(unsigned y, unsigned x);
  /// If any of x,y,z is zero then all become
  /// zero and the Dim is unknown
  Dim(unsigned z, unsigned y, unsigned x);
  Dim(const Dim &other);
  Dim(const Dim &&other);
  virtual ~Dim() = default;

public:
  virtual bool operator==(const Dim &other) const;
  virtual bool operator!=(const Dim &other) const;
  virtual operator unsigned long long() const;
  virtual Dim &operator=(const Dim &other);

  /// Index 0 refers to the x-dimension, 1 to the y-dimension and 2 to the
  /// z-dimension.
  /// @throw std::out_of_range {}
  virtual unsigned& operator[](unsigned int idx);

  /// Index 0 refers to the x-dimension, 1 to the y-dimension and 2 to the
  /// z-dimension.
  /// @throw std::out_of_range {}
  virtual unsigned operator()(unsigned int idx) const;

  ///
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Dim &dim);

  ///
  friend std::ostream &operator<<(std::ostream &os, const Dim &dim);

public:
  /// Check if unknown
  bool isUnknown() const;

  /// Retrieve the size of this dim
  unsigned long long size() const;

  /// Check if the dim is 1-dimensional,
  /// i.e spans only in the x-dimension
  bool is1D() const;

  /// Check if the dim is 2-dimensional,
  /// i.e spans in the y-dimension (x-dimension may still be 1)
  bool is2D() const;

  /// Check if the dim is 3-dimensional,
  /// i.e spans into the z-dimensions (x and y dimensions may still be 1)
  bool is3D() const;

  /// Check if the dim grows at most in 1 dimension
  /// The difference compared to is1D() is that y and z
  /// are also considered. Example:
  /// \code
  /// Dim dimA(1);     // is1D and isEffective1D
  /// Dim dimB(1,2);   // is2D and isEffective1D
  /// Dim dimC(1,1,2); // is3D and isEffective1D
  /// \endcode
  bool isEffective1D() const;

  /// Check if the dim grows exactly in 2 dimensions
  /// The difference compared to is2D() is that any
  /// combination is considered. Example:
  /// \code
  /// Dim dimA(1,2,1); // is2D and isEffective1D
  /// Dim dimB(1,2,2); // is3D and isEffective2D
  /// Dim dimC(3,1,3); // is3D and isEffective2D
  /// \endcode
  bool isEffective2D() const;

  /// Check if an index is valid index for this dim
  bool hasIndex(const Index &idx) const;
  bool hasIndex(unsigned long long linearIdx) const;

  /// Check if a linear index is valid index for this dim
  /// The index is first delinearized based on the dim's
  /// values.
  bool hasLinearIndex(unsigned long long idx) const;

  /// Get the minimum index for this dim
  /// (0,0,0) unless overriden
  virtual Index getMinIndex() const;

  /// Get the maximum index for this dim (inclusive).
  /// (Z-1,Y-1,X-1) unless overriden
  virtual Index getMaxIndex() const;

  /// Get the mimimum linear index for this dim
  /// 0 unless overriden
  virtual unsigned long long getMinLinearIndex() const;

  /// Get the maximum linear index for this dim
  virtual unsigned long long getMaxLinearIndex() const;

  virtual unsigned int cols() const;
  virtual unsigned int rows() const;
  virtual unsigned int layers() const;

  virtual unsigned int width() const;
  virtual unsigned int height() const;
  virtual unsigned int depth() const;

  virtual std::string toString() const;

public:
  unsigned int x, y, z;

public:
  /// Special DIM used to indicated erroneous 
  /// dim or  unknown dim
  static Dim None;

  /// The unit dim
  static Dim Unit;

  static Dim Linear2;
  static Dim Linear4;
  static Dim Linear8;
  static Dim Linear16;
  static Dim Linear32;
  static Dim Linear64;
  static Dim Linear128;
  static Dim Linear256;
  static Dim Linear512;
  static Dim Linear1024;
  static Dim Square2;
  static Dim Square4;
  static Dim Square8;
  static Dim Square16;
  static Dim Square32;
  static Dim Square64;
  static Dim Square128;
  static Dim Square256;
  static Dim Square512;
  static Dim Square1024;
  static Dim Cube2;
  static Dim Cube4;
  static Dim Cube8;
  static Dim Cube16;
  static Dim Cube32;
  static Dim Cube64;
  static Dim Cube128;
  static Dim Cube256;
  static Dim Cube512;
  static Dim Cube1024;
};

} // namespace kerma

#endif