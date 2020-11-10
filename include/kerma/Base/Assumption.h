#ifndef KERMA_BASE_ASSUMPTION_H
#define KERMA_BASE_ASSUMPTION_H

#include "kerma/Base/Dim.h"
#include "kerma/Base/Memory.h"
#include <llvm-10/llvm/IR/DerivedTypes.h>
#include <llvm-10/llvm/IR/Value.h>
#include <llvm-10/llvm/Support/raw_ostream.h>
#include <ostream>

namespace kerma {

class Assumption {
public:
  enum AssumptionKind {
    AK_DIM=0,
    AK_VAL,
    AK_IVAL,
    AK_FPVAL
  };

private:
  AssumptionKind Kind;
protected:
  Assumption(AssumptionKind K): Assumption(K, nullptr) {}
  Assumption(AssumptionKind K, llvm::Value *V) : Kind(K), V(V) {}
  llvm::Value *V = nullptr;

public:
  Assumption(const Assumption &) = delete;
  AssumptionKind getKind() const { return Kind; }
  // virtual Assumption& operator=(const Assumption& other)=0;
  virtual bool operator==(const Assumption &O) { return V == O.V; }
  virtual bool operator!=(const Assumption &O) { return !operator==(O); }
  virtual llvm::Value *getIRValue() { return V; }
  virtual void setIRValue(llvm::Value *V) { this->V = V;}
  virtual void print(llvm::raw_ostream &OS) { OS << "WTF"; }
};


llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Assumption &M);
std::ostream &operator<<(std::ostream &OS, Assumption &M);

class DimAssumption : public Assumption {
private:
  Dim D;
  Memory *M;
public:
  DimAssumption(Dim D) : Assumption(AK_DIM), D(D) {}
  DimAssumption(Dim D, Memory &M) : Assumption(AK_DIM), D(D), M(&M) {
    V = M.getValue();
  }
  static bool classof(const Assumption *A);
  Dim& getDim() { return D; }
  Memory *getMemory() { return M; }
  DimAssumption &setMemory(Memory &M) {
    this->M = &M;
    V = M.getValue();
    return *this;
  }
  virtual void print(llvm::raw_ostream &OS) override {
    OS << "shape " << D << " for ";
    if ( V) OS << *V; else OS << "nullptr";
  }
};

class ValAssumption : public Assumption {
protected:
  ValAssumption() : Assumption(AK_DIM) {}
  ValAssumption(llvm::Value *V) : Assumption(AK_VAL, V) {}
  ValAssumption(AssumptionKind Kind, llvm::Value *V) : Assumption(Kind, V) {}
public:
  static bool classof(const Assumption *A);
};

class IAssumption : public ValAssumption {
private:
  long long Val;
public:
  IAssumption(long long Val) : IAssumption(Val,nullptr) {}
  IAssumption(long long Val, llvm::Value *V) : ValAssumption(AK_IVAL, V), Val(Val) {}
  llvm::Value *getIRValue() override { return V; }
  long long getValue() { return Val; }
  static bool classof(const Assumption *A);
  virtual void print(llvm::raw_ostream &OS) override {
    OS << "value " << Val << " for ";
    if ( V) OS << *V; else OS << "nullptr";
  }
};

class FPAssumption : public ValAssumption {
private:
  double Val;
public:
  FPAssumption(double Val) : FPAssumption(Val,nullptr) {}
  FPAssumption(double Val, llvm::Value *V) : ValAssumption(AK_FPVAL, V), Val(Val) {}
  llvm::Value *getIRValue() override { return V; }
  double getValue() { return Val; }
  static bool classof(const Assumption *A);
  virtual void print(llvm::raw_ostream &OS) override {
    OS << "value " << Val << " for ";
    if ( V) OS << *V; else OS << "nullptr";
  }
};

} // namespace kerma

#endif // KERMA_BASE_ASSUMPTION_H