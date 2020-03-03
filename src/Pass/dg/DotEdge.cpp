#include <kerma/passes/dg/Dot.h>

#include <string>
#include <utility>

using namespace kerma;

DotEdge::DotEdge(DotNode &src, DotNode &tgt)
: src_(src), tgt_(tgt)
{}

DotEdge::DotEdge(const DotEdge& other) : src_(other.src_), tgt_(other.tgt_)
{}

DotNode& DotEdge::getSource() { return src_; }
DotNode& DotEdge::getTarget() { return tgt_; }

std::string DotEdge::getValue() {
  return src_.getName() + " -> " + tgt_.getName();
}

DotEdge& DotEdge::operator=(DotEdge& other)
{
  src_ = other.src_;
  tgt_ = other.tgt_;
  return *this;
}

bool DotEdge::operator<(const DotEdge& other) const
{ 
  return std::make_pair(src_, tgt_) < std::make_pair(other.src_, other.tgt_);
}

bool DotEdge::operator==(const DotEdge& other)
{
  return src_ == other.src_ && tgt_ == other.tgt_;
}

bool DotEdge::operator!=(const DotEdge& other)
{
  return !operator==(other);
}

std::ostream& operator<<(std::ostream& os, DotEdge& e)
{
  os << e.getSource().getName() << " -> " << e.getTarget().getName();
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& ros, DotEdge& e)
{
  ros << e.getSource().getName() << " -> " << e.getTarget().getName();
  return ros;
}