//===-- kerma/GenericOpts.h - Common command line opts ---------*- C++ -*--===//
//
// Part of the kerma project
//
//===----------------------------------------------------------------------===//
//
// This file defines some command line options that are common among multiple
// passes. For instance some passes may require as input the host side LLVM IR, 
// or we may want verbose output etc. If each pass defined ALL the CL opts it
// needs, then if there are conflicts (e.g two passes register the same cl opt)
// the opt tool will issue a warning and not run the pass since it expects cl
// opts to be unique. Thus this file groups the common options to avoid this
// issue
//
//===----------------------------------------------------------------------===//
#ifndef KERMA_PASS_GENERIC_OPTS_H
#define KERMA_PASS_GENERIC_OPTS_H

#include <llvm/Support/CommandLine.h>

namespace kerma
{
namespace cl
{

extern bool KermaDebugFlag;

} // namespace cl
} // namespace kerma


#endif // KERMA_PASS_GENERIC_OPTS_H