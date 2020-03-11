#ifndef KERMA_SUPPORT_TERMINAL_H
#define KERMA_SUPPORT_TERMINAL_H

#include "llvm/Support/raw_ostream.h"
#include <istream>
#include <ostream>
#include <sstream>

namespace kerma
{
namespace term
{


#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"

#define RST "\x1B[0m"
#define BOLD "\x1B[1m"
#define UNDL "\x1B[4m"


namespace detail
{
  
struct FmtTermStringStream : std::stringstream 
{
public:
  FmtTermStringStream();
  FmtTermStringStream(const FmtTermStringStream &ftss);
  ~FmtTermStringStream()=default;

  FmtTermStringStream & operator << (FmtTermStringStream &ts);
  FmtTermStringStream & operator << (const char *s);

  friend std::ostream & operator << (std::ostream &out, const FmtTermStringStream &ts);
  friend llvm::raw_ostream & operator << (llvm::raw_ostream &out, const FmtTermStringStream &ts);
};

} // namespace detail

std::ostream & operator<<(std::ostream &out, std::stringstream &ss);
llvm::raw_ostream & operator << (llvm::raw_ostream &out, std::stringstream &ss);

detail::FmtTermStringStream bold(const detail::FmtTermStringStream &ftss);
detail::FmtTermStringStream bold(const char *s);

detail::FmtTermStringStream underline(const detail::FmtTermStringStream &ftss);
detail::FmtTermStringStream underline(const char *s);

detail::FmtTermStringStream red(const detail::FmtTermStringStream &ftss);
detail::FmtTermStringStream red(const char *s);

detail::FmtTermStringStream blue(const detail::FmtTermStringStream &ftss);
detail::FmtTermStringStream blue(const char *s);

} // namespace term

} /// namespace kerma
#endif