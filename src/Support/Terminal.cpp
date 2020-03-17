//===-- Support/Terminal.cpp ----------------------------------------------===//
//
// Part of the kerma project
//
//===----------------------------------------------------------------------===//
//
// This file implements kerma/Support/Terminal.h
// TODO \todo Needs testing
// 
//===----------------------------------------------------------------------===//
#include "llvm/Support/raw_ostream.h"
#include <kerma/Support/Terminal.h>

#include <unistd.h>
#include <sstream>
#include <string>

namespace kerma
{

namespace term
{

/// TODO Improve robustness
const bool termHasColor() 
{
  // Taken from loguru:
  // https://github.com/emilk/loguru/blob/master/loguru.cpp#L219
  if (!isatty(STDERR_FILENO))
    return false;

  if (const char* term = getenv("TERM"))
    return 0 == strcmp(term, "cygwin")
        || 0 == strcmp(term, "linux")
        || 0 == strcmp(term, "rxvt-unicode-256color")
        || 0 == strcmp(term, "screen")
        || 0 == strcmp(term, "screen-256color")
        || 0 == strcmp(term, "screen.xterm-256color")
        || 0 == strcmp(term, "tmux-256color")
        || 0 == strcmp(term, "xterm")
        || 0 == strcmp(term, "xterm-256color")
        || 0 == strcmp(term, "xterm-termite")
        || 0 == strcmp(term, "xterm-color");
  else
    return false;
};

/// TODO Improve robustness
const bool termIsXterm()
{
  if (const char* term = getenv("TERM")) {
    return 0 == strcmp(term, "xterm")
        || 0 == strcmp(term, "xterm-256color")
        || 0 == strcmp(term, "xterm-termite")
        || 0 == strcmp(term, "xterm-color");
  }
  return false;
}

/// TODO Improve robustness
const bool termInTmux()
{
  if (const char *term = getenv("TERM"))
    return std::string(term).find("tmux") != std::string::npos;
}


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

