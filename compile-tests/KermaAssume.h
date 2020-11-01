#include <type_traits>
#include <assert.h>

struct KermaAssumeDim {
  KermaAssumeDim(const char *Kernel, unsigned int pos, unsigned int x) {}
  KermaAssumeDim(const char *Kernel, unsigned int pos, unsigned int y, unsigned int x) {}
  KermaAssumeDim(const char *Kernel, unsigned int pos, unsigned int z, unsigned int y, unsigned int x) {}
};

template<typename T>
struct KermaAssumeVal {
  KermaAssumeVal(const char *Kernel, unsigned int pos, T t) {
    assert(std::is_fundamental<T>::value);
  }
};

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE(x) CONCATENATE(x, __COUNTER__)

#define KERMA_ASSUME_1D( Kernel, Pos, X) static KermaAssumeDim MAKE_UNIQUE(Assumption)(Kernel, Pos, X)