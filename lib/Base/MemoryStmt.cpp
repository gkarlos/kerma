#include "kerma/Base/MemoryStmt.h"
#include "kerma/Base/Loop.h"
#include "kerma/Base/Memory.h"
#include "kerma/Base/MemoryAccess.h"
#include "kerma/SourceInfo/SourceRange.h"
#include <mutex>
#include <ostream>

using namespace llvm;
namespace kerma {

static std::mutex mtx;

static unsigned int genID() {
  static volatile unsigned int IDs = 0;
  unsigned int id;
  mtx.lock();
  id = IDs++;
  mtx.unlock();
  return id;
}

MemoryStmt::MemoryStmt(SourceRange R, Type Ty, KermaNode *Parent)
    : MemoryStmt(genID(), R, Ty, Parent) {}
MemoryStmt::MemoryStmt(unsigned int ID, SourceRange R, Type Ty,
                       KermaNode *Parent)
    : KermaNode(NK_MemStmt, R, Parent), ID(ID), Ty(Ty) {}

MemoryStmt &MemoryStmt::setRange(const SourceRange &R) {
  this->R = R;
  if (!R) {
    MAS.clear();
  } else {
    auto NotInRange = [&R](const MemoryAccess &MA) {
      return !R.contains(MA.getLoc()) && !R.containsLine(MA.getLoc());
    };
    std::remove_if(MAS.begin(), MAS.end(), NotInRange);
  }
  return *this;
}

bool MemoryStmt::addMemoryAccess(MemoryAccess &MA, SourceInfo &SI) {
  if (!R) {
    if (auto &Range = SI.getRangeForLoc(MA.getLoc())) {
      R = Range;
    } else {
      return false;
    }
  }

  if (R.contains(MA.getLoc()) || R.containsLine(MA.getLoc())) {
    MAS.push_back(MA);
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

static std::string tystr(MemoryStmt::Type Ty) {
  if (Ty == MemoryStmt::RD)
    return "R";
  else if (Ty == MemoryStmt::WR)
    return "W";
  else if (Ty == MemoryStmt::RDWR)
    return "RW";
  else
    return "U";
}

void MemoryStmt::print(llvm::raw_ostream &O) const {
  auto nesting = std::string(getNesting(), '\t');
  auto nestingPlus1 = nesting + '\t';
  O << std::string(getNesting(), '\t') << '(' << tystr(getType()) << ") " << getRange() << " { ";
  for (auto &MA : getAccesses())
    O << "#" << MA.getID() << ' ';
  O << "} #" << getID();
  if ( getParent()) {
    O << " parent: #" << dyn_cast<LoopNest>(getParent())->getID();
  } else {
    O << " parent: none";
  }
  for ( auto &A : this->getAccesses())
    O << '\n' << nestingPlus1 << A;
}


bool MemoryStmt::classof(const KermaNode *S) {
  return S->getKind() == NK_MemStmt;
}


} // namespace kerma