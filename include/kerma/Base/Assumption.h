#ifndef KERMA_BASE_ASSUMPTION_H
#define KERMA_BASE_ASSUMPTION_H

#include "kerma/Base/Dim.h"
// #include "kerma/Base/Kernel.h"
#include "kerma/Base/Memory.h"
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>
#include <ostream>

namespace kerma {

class Assumption {
public:
  enum AssumptionKind { AK_DIM = 0, AK_LAUNCH, AK_VAL, AK_IVAL, AK_FPVAL };

private:
  AssumptionKind Kind;

protected:
  Assumption(AssumptionKind K) : Assumption(K, nullptr) {}
  Assumption(AssumptionKind K, llvm::Value *V) : Kind(K), V(V) {}
  Assumption(const Assumption &O) { *this = O; }
  llvm::Value *V = nullptr;

public:
  AssumptionKind getKind() const { return Kind; }
  virtual Assumption &operator=(const Assumption &O) {
    V = O.V;
    return *this;
  }
  virtual bool operator==(const Assumption &O) {
    return Kind == O.Kind && V == O.V;
  }
  virtual bool operator!=(const Assumption &O) { return !operator==(O); }
  virtual llvm::Value *getIRValue() { return V; }
  virtual void setIRValue(llvm::Value *V) { this->V = V; }
  virtual void print(llvm::raw_ostream &OS) const {
    // OS << "assumption for ";
    // if (V)
    //   OS << *V;
    // else
    //   OS << "nullptr";
    OS << "none";
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Assumption &M);
std::ostream &operator<<(std::ostream &OS, Assumption &M);

class DimAssumption : public Assumption {
private:
  Dim D;
  Memory *M;

public:
  DimAssumption() : DimAssumption(Dim::None) {}
  DimAssumption(Dim D) : Assumption(AK_DIM), D(D) {}
  DimAssumption(Dim D, Memory &M)
      : Assumption(AK_DIM, M.getValue()), D(D), M(&M) {}
  DimAssumption(const DimAssumption &O) : Assumption(O) { *this = O; }
  Dim &getDim() { return D; }
  Memory *getMemory() { return M; }
  DimAssumption &setMemory(Memory &M) {
    this->M = &M;
    setIRValue(M.getValue());
    return *this;
  }

  virtual bool operator==(const DimAssumption &O) {
    return Assumption::operator==(O) && M == O.M && D == O.D;
  }
  virtual bool operator!=(const DimAssumption &O) { return !operator==(O); }
  virtual DimAssumption &operator=(const DimAssumption &O) {
    Assumption::operator=(O);
    D = O.D;
    M = O.M;
    return *this;
  }
  virtual void print(llvm::raw_ostream &OS) const override {
    OS << D;
    // OS << "shape " << D << " for ";
    // if (V)
    //   OS << *V;
    // else
    //   OS << "nullptr";
  }
  static bool classof(const Assumption *A);
};

class LaunchAssumption : public Assumption {
private:
  Dim BlockDim;
  Dim GridDim;
  llvm::Function *Kernel;

public:
  LaunchAssumption() : LaunchAssumption(Dim::None, Dim::None, nullptr) {}
  LaunchAssumption(const Dim &Grid, const Dim &Block, llvm::Function *Kernel)
      : Assumption(AK_LAUNCH, Kernel), GridDim(Grid), BlockDim(Block),
        Kernel(Kernel) {}
  LaunchAssumption(const LaunchAssumption &O) : Assumption(O) { *this = O; }
  const Dim &getBlock() const { return BlockDim; }
  const Dim &getGrid() const { return GridDim; }
  LaunchAssumption &setKernel(llvm::Function &F) {
    Kernel = &F;
    setIRValue(&F);
    return *this;
  }

  llvm::Function *getKernel() { return Kernel; }
  const Dim &getGrid() { return GridDim; }
  const Dim &getBlock() { return BlockDim; }

  virtual bool operator==(const LaunchAssumption &O) {
    return Assumption::operator==(O) && BlockDim == O.BlockDim &&
           GridDim == O.GridDim;
  }
  virtual bool operator!=(const LaunchAssumption &O) { return !operator==(O); }
  virtual LaunchAssumption &operator=(const LaunchAssumption &O) {
    Assumption::operator=(O);
    BlockDim = O.BlockDim;
    GridDim = O.GridDim;
    Kernel = O.Kernel;
    return *this;
  }

  virtual void print(llvm::raw_ostream &OS) const override {
    OS << GridDim << ',' << BlockDim;
    // OS << "launch " << GridDim << " | " << BlockDim << " for ";
    // if (Kernel)
    //   OS << Kernel->getName();
    // else
    //   OS << "nullptr";
  }

  static bool classof(const Assumption *A);
};

class ValAssumption : public Assumption {
public:
  ValAssumption() : Assumption(AK_DIM) {}
  ValAssumption(llvm::Value *V) : Assumption(AK_VAL, V) {}
  ValAssumption(AssumptionKind Kind, llvm::Value *V) : Assumption(Kind, V) {}
  ValAssumption(const ValAssumption &O) : Assumption(O) {}
  ValAssumption &operator=(const ValAssumption &O) {
    Assumption::operator=(O);
    return *this;
  }
  virtual bool operator==(const ValAssumption &O) {
    return Assumption::operator==(O);
  }
  virtual bool operator!=(const ValAssumption &O) { return !operator==(O); }
  // virtual void print(llvm::raw_ostream &OS) const override {
  //   ValAssumption::print(OS);
  // }
public:
  static bool classof(const Assumption *A);
};

class IAssumption : public ValAssumption {
private:
  long long Val;

public:
  IAssumption(long long Val) : IAssumption(Val, nullptr) {}
  IAssumption(long long Val, llvm::Value *V)
      : ValAssumption(AK_IVAL, V), Val(Val) {}
  llvm::Value *getIRValue() override { return V; }
  long long getValue() { return Val; }

  virtual bool operator==(const IAssumption &O) {
    return Assumption::operator==(O) && Val == O.Val;
  }

  virtual bool operator!=(const IAssumption &O) { return !operator==(O); }

  virtual IAssumption &operator=(const IAssumption &O) {
    Assumption::operator=(O);
    Val = O.Val;
    return *this;
    return *this;
  }

  virtual void print(llvm::raw_ostream &OS) const override {
    OS << Val;
  }

  static bool classof(const Assumption *A);
};

class FPAssumption : public ValAssumption {
private:
  double Val;

public:
  FPAssumption(double Val) : FPAssumption(Val, nullptr) {}
  FPAssumption(double Val, llvm::Value *V)
      : ValAssumption(AK_FPVAL, V), Val(Val) {}
  llvm::Value *getIRValue() override { return V; }
  double getValue() { return Val; }

  virtual bool operator==(const FPAssumption &O) {
    return Assumption::operator==(O) && Val == O.Val;
  }

  virtual bool operator!=(const FPAssumption &O) { return !operator==(O); }

  virtual FPAssumption &operator=(const FPAssumption &O) {
    Assumption::operator=(O);
    Val = O.Val;
    return *this;
  }

  // virtual void print(llvm::raw_ostream &OS) override {
  //   OS << "value " << Val << " for ";
  //   if (V)
  //     OS << *V;
  //   else
  //     OS << "nullptr";
  // }

  virtual void print(llvm::raw_ostream &OS) const override {
    OS << Val;
  }

  static bool classof(const Assumption *A);
};

} // namespace kerma

#endif // KERMA_BASE_ASSUMPTION_H