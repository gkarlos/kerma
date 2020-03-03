#ifndef KERMA_SUPPORT_PRETTYPRINTABLE_H
#define KERMA_SUPPORT_PRETTYPRINTABLE_H


#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <ostream>

namespace kerma
{

/*
 * Abstract Base Class denoting that derived classes must be "pretty-printable" to llvm::raw_ostream
 */
class LLVMPrettyPrintable
{
  virtual void pp(llvm::raw_ostream &os) = 0;
};

/*
 * Abstract Base Class denoting that derived classes must be "pretty-printable" to std::ostream
 */
class STDPrettyPrintable
{
  virtual void pp(std::ostream &os) = 0;
};

/*
 *  Abstract Base Class denoting that derived classes must be "pretty-printable" to
 *  both llvm::raw_ostream and std::ostream
 */
class PrettyPrintable : public LLVMPrettyPrintable, public STDPrettyPrintable
{};

/*
 * Abstract Base Class denoting that derived classes must be "pretty-printable" to llvm::raw_fd_ostream
 */
class LLVMFilePrettyPrintable
{
  virtual void pp(llvm::raw_fd_ostream &fs) = 0;
};

/*
 * Abstract Base Class denoting that derived classes must be "pretty-printable" to std::ofstream
 */
class STDFilePrettyPrintable
{
  virtual void pp(std::ofstream &fs) = 0;
};

/*
 * Abstract Base Class denoting that derived classes must be "pretty-printable" to
 * both llvm::raw_fd_ostream and std::ofstream
 */
class FilePrettyPrintable : public LLVMFilePrettyPrintable, STDFilePrettyPrintable
{
  
};

} /* NAMESPACE kerma */

#endif /* KERMA_SUPPORT_PRETTYPRINTABLE_H */