#include "llvm/Support/raw_ostream.h"
#include <kerma/Support/Terminal.h>
#include <sstream>

namespace kerma
{

namespace term
{

namespace detail
{

FmtTermStringStream::FmtTermStringStream() : std::stringstream()
{}

FmtTermStringStream::FmtTermStringStream(const FmtTermStringStream &ftss)
: FmtTermStringStream()
{
  *this << ftss.str();
}

FmtTermStringStream & 
FmtTermStringStream::operator << (FmtTermStringStream &ftss)
{
  *(static_cast<std::stringstream*>(this)) << ftss.str();
  return *this;
}

FmtTermStringStream &
FmtTermStringStream::operator << (const char *s)
{
  *(static_cast<std::stringstream*>(this)) << s;
  return *this;
}

std::ostream & operator<<(std::ostream &out, const detail::FmtTermStringStream &ftss) 
{
  out << ftss.str();
  return out;
}

llvm::raw_ostream & operator<<(llvm::raw_ostream &ros, const detail::FmtTermStringStream &ftss)
{
  ros << ftss.str();
  return ros;
}

} // namespace detail

detail::FmtTermStringStream bold(const detail::FmtTermStringStream &ftss)
{
  detail::FmtTermStringStream ss;
  ss << BOLD << ftss.str() << RST;
  return ss;
}

detail::FmtTermStringStream bold(const char *m) 
{
  detail::FmtTermStringStream ss;
  ss << BOLD << m << RST;
  return ss;
}

detail::FmtTermStringStream underline(const detail::FmtTermStringStream &ftss)
{
  detail::FmtTermStringStream ss;
  ss << UNDL << ftss.str() << RST;
  return ss;
}

detail::FmtTermStringStream underline(const char *s)
{
  detail::FmtTermStringStream ss;
  ss << UNDL << ss.str() << RST;
  return ss;
}

detail::FmtTermStringStream red(const detail::FmtTermStringStream &ftss)
{
  detail::FmtTermStringStream ss;
  ss << RED << ftss.str() << RST;
  return ss;
}

detail::FmtTermStringStream red(const char *m) 
{
  detail::FmtTermStringStream ss;
  ss << RED << m << RST;
  return ss;
}

detail::FmtTermStringStream blue(const detail::FmtTermStringStream &ftss)
{
  detail::FmtTermStringStream ss;
  ss << BLU << ftss.str() << RST;
  return ss;
}

detail::FmtTermStringStream blue(const char *m) 
{
  detail::FmtTermStringStream ss;
  ss << BLU << m << RST;
  return ss;
}

} // namespace term

} // namespace kerma

