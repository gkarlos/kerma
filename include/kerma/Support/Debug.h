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

#endif //KERMA_SUPPORT_DEBUG_H