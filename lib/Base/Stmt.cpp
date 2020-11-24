#include "kerma/Base/Stmt.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/Memory.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <mutex>
#include <ostream>

using namespace llvm;
namespace kerma {

// static std::mutex mtx;

// static unsigned int genID() {
//   static volatile unsigned int IDs = 0;
//   unsigned int id;
//   mtx.lock();
//   id = IDs++;
//   mtx.unlock();
//   return id;
// }
// Stmt::Stmt(SourceRange R, Type Ty, KermaNode *Parent)
// //     : Stmt(genID(), R, Ty, Parent) {}
// // Stmt::Stmt(unsigned int ID, SourceRange R, Type Ty,
// //                        KermaNode *Parent)
//     : KermaNode(NK_MemStmt, R, Parent), Ty(Ty) {}

Stmt &Stmt::setRange(const SourceRange &R) {
  this->R = R;
  if (!R) {
    Accesses.clear();
  } else {
    auto NotInRange = [&R](const MemoryAccess &MA) {
      return !R.contains(MA.getLoc()) && !R.containsLine(MA.getLoc());
    };
    std::remove_if(Accesses.begin(), Accesses.end(), NotInRange);
  }
  return *this;
}

bool Stmt::addMemoryAccess(MemoryAccess &MA, SourceInfo &SI) {
  if (!R) {
    if (auto &Range = SI.getRangeForLoc(MA.getLoc())) {
      R = Range;
    } else {
      return false;
    }
  }

  if (R.contains(MA.getLoc()) || R.containsLine(MA.getLoc())) {
    Accesses.push_back(MA);
    switch (MA.getType()) {
    case MemoryAccess::Type::Load: {
      if (this->Ty == UKN)
        this->Ty = RD;
      else
        this->Ty = (this->Ty == WR) ? RDWR : this->Ty;
    } break;
    case MemoryAccess::Type::Store:
    case MemoryAccess::Type::Memset:
      if (this->Ty == UKN)
        this->Ty = WR;
      else
        this->Ty = (this->Ty == RD) ? RDWR : this->Ty;
      break;
    case MemoryAccess::Type::Memcpy:
    case MemoryAccess::Type::Memmove:
      this->Ty = RDWR;
    default:
      break;
    }
    return true;
  }
  return false;
}

static std::string tystr(Stmt::Type Ty) {
  if (Ty == Stmt::RD)
    return "RD";
  else if (Ty == Stmt::WR)
    return "WR";
  else if (Ty == Stmt::RDWR)
    return "R|W";
  else
    return "U";
}

void Stmt::print(llvm::raw_ostream &O) const {
  auto nesting = std::string(getNesting(), '\t');
  auto nestingPlus1 = std::string(getNesting() + 1, '\t');
  O << nesting << '(' << tystr(getType()) << ") " << getRange() << " #"
    << this->getID() << " { ";
  for (auto &MA : getAccesses())
    O << "#" << MA.getID() << ' ';
  O << "} .";
  if ( getParent()) {
    O << " parent: #" << getParent()->getID();
  } else {
    O << " parent: none";
  }
  for ( auto &A : this->getAccesses())
    O << '\n' << nestingPlus1 << A;
}


bool Stmt::classof(const KermaNode *S) {
  return S->getKind() == NK_Stmt;
}


} // namespace kerma