#ifndef KERMA_SUPPORT_UTIL_H
#define KERMA_SUPPORT_UTIL_H

#include <iostream>

#define ERROR(msg, fatal) \
  do { \
      std::cerr << "ERROR" << ((fatal)? " (fatal)" : "") << " at " << __FILE__ << ":" << __LINE__ << ", function " << __func__ << "():\n    "; \
      std::cerr << msg << std::endl; \
      if ( fatal) \
        abort(); \
  } while (false)

#define FATAL(msg) ERROR(msg, true)

#define NOT_IMPLEMENTED_YET FATAL("not implemented yet")

#endif /* KERMA_SUPPORT UTIL_H */