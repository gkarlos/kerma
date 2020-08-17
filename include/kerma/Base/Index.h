#ifndef KERMA_BASE_INDEX_H
#define KERMA_BASE_INDEX_H

namespace kerma {

class Dim;

class Index {
public:
 Index(unsigned int z, unsigned int y, unsigned int x);
 Index(unsigned int y, unsigned int x);
 Index(unsigned int x);
 Index(const Index &other);
 Index(const Index &&other);
 virtual ~Index();

public:
  static unsigned int linearize(const Index &idx, const Dim &dim);

  static Index delinearize(unsigned int idx, const Dim &dim);

public:
  unsigned int x;
  unsigned int y;
  unsigned int z;

  static Index None;
};

} // namespace kerma

#endif