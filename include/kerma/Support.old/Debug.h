#ifndef KERMA_SUPPORT_DEBUG_H
#define KERMA_SUPPORT_DEBUG_H

#include <kerma/GenericOpts.h>

#ifdef NDEBUG
  #define KERMA_DEBUG(X)
#else
  #define KERMA_DEBUG(X)               \
    do {                               \
      if ( kerma::cl::KermaDebugFlag)  \
      {                                \
        X;                             \
      }                                \
    } while ( 0 )
#endif

namespace kerma
{
namespace detail
{

inline static
std::string className(const std::string& prettyFunction)
{
    size_t colons = prettyFunction.rfind("("); // skip arguments
           colons = prettyFunction.substr(0,colons).rfind("::"); // skip fn function
           
    if (colons == std::string::npos)
        return "::";

    size_t begin = prettyFunction.substr(0,colons).rfind(" ") + 1; // skip fn signature
    size_t end = colons - begin;

    return prettyFunction.substr(begin,end);
}

} 
}

#define __CLASS_NAME__ kerma::detail::className(__PRETTY_FUNCTION__)

#endif //KERMA_SUPPORT_DEBUG_H