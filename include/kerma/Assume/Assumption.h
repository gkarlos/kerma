#ifndef KERMA_BASE_ASSUMPTION_H
#define KERMA_BASE_ASSUMPTION_H

#include "kerma/Base/Dim.h"

#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>


namespace kerma {

/// Base (abstract) class for all Assumptions
class Assumption {
public:
  enum AssumptionKind {
    AK_DIM,
    AK_ARGDIM,
    AK_VAL
  };

private:
  AssumptionKind Kind;

protected:
  Assumption()=default;

public:
  AssumptionKind getKind() const { return Kind; }
  virtual Assumption& operator=(const Assumption& other);
  virtual bool operator==(const Assumption& other);
};

class DimAssumption : public Assumption {
private:
  llvm::Value* Value;
  Dim Dim;
public:
  DimAssumption(){}
  // llvm rtti
  static bool classof(const Assumption *A) {
    return A->getKind() <= AK_ARGDIM;
  }
};

class ArgDimAssumption : public DimAssumption {

  static bool classof(const Assumption *A) {
    return A->getKind() == AK_ARGDIM;
  }
};


class ValAssumption : public Assumption {
public:
  ValAssumption()=default;


  // llvm rrti
  static bool classof(const Assumption *A) {
    return A->getKind() == AK_VAL;
  }
};


} // namespace kerma

#endif // KERMA_BASE_ASSUMPTION_H