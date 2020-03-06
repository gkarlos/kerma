/*
 * @file: include/Cuda/CudaKernel.h
 */
#ifndef KERMA_SUPPORT_CUDA_H
#define KERMA_SUPPORT_CUDA_H

#include "kerma/Support/PrettyPrintable.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include <kerma/Cuda/Cuda.h>

namespace kerma
{

class CudaKernel;
class CudaKernelLaunch;
class CudaKernelLaunchConfiguration;

class CudaKernel : public PrettyPrintable
{

private:
  llvm::Function &fn_;
  CudaSide IRModuleSide_;
  std::string name_;
  std::string mangledName_;
  unsigned int signatureLineStart_;
  unsigned int signatureLineEnd_;
  unsigned int bodyLineStart_;
  unsigned int bodyLineEnd_;

public:
  CudaKernel(llvm::Function &fn, CudaSide IRModuleSide);
  CudaKernel(llvm::Function &fn);
  ~CudaKernel()=default;

public:
  bool operator==(const CudaKernel &other) const;
  bool operator<(const CudaKernel &other) const;
  bool operator>(const CudaKernel &other) const;

public:
  virtual void pp(llvm::raw_ostream& os);
  virtual void pp(std::ostream& os);

public:
  /*
   * Get a pointer to the llvm::Function for this kernel
   */
  llvm::Function &getFn();

  /*
   * Set the side of the LLVM IR file that this kernel was detected in (host or device)
   */
  void setIRModuleSide(CudaSide IRModuleSide);

  /*
   * Retrieve the side of the LLVM IR file this kernel was detected at
   */
  CudaSide getIRModuleSide();

  /*
   * Retrieve the number of arguments of this kernel
   */
  int getNumArgs();

  /*
   * Retrieve the name of this kernel
   */
  std::string& getName();

  /*
   * Retrieve the mangled name of this kernel
   */
  std::string& getMangledName();

  /*
   * @brief Set the line number at which the signature of this CudaKernel's definition starts
   * 
   * if @p line > @pre #signatureLineEnd_, then @post #signatureLineEnd_ is set to #SRC_LINE_UNKNOWN
   *
   * @post signatureLineStart_ <= signatureLineEnd_ || signatureLineEnd_ == SRC_LINE_UNKNOWN
   */
  void setSignatureLineStart(unsigned int line);

  /*
   * @brief Set the line number at which the signature of this CudaKernel's definition ends
   *
   * if @p line < @pre #signatureLineStart_, then @post #signatureLineEnd_ is set to #SRC_LINE_UNKNOWN
   *
   * @post signatureLineStart_ <= signatureLineEnd_ || signatureLineEnd_ == SRC_LINE_UNKNOWN
   */
  void setSignatureLineEnd(unsigned int line);

  /*
   * @brief Set the start line and end line of this CudaKernel's signature in source code
   *        Equivalent to calling #setSignatureLineStart() followed by #setSignatureLineEnd()
   */
  void setSignatureLines(unsigned int start, unsigned int end);

  /*
   * @brief Retrieve the start line of this CudaKernel's signature in source code
   * @return An integer >= 0 or SRC_LINE_UNKNOWN
   */
  int getSignatureLineStart();

  /*
   * @brief Retrieve the end line of this CudaKernel's signature in source code
   * @return An integer >= 0 or SRC_LINE_UNKNOWN
   */
  int getSignatureLineEnd();

  /*
   * @brief Retrieve the number of lines this CudaKernel's signature spans over (inclusive)
   * 
   * The number of lines is calculated as <code> 1 + #signatureLineEnd_ - #signatureLineStart_ </code>
   *
   * If any of #signatureLineStart_ or #signatureLineEnd_ have the value #SRC_LINE_UNKNOWN, then the
   * number of lines cannot be calculated and 0 is returned.
   * 
   * @return A positive integer or 0
   */
  int getSignatureNumLines();

  /*
   * @brief Set the line number of which the body of this CudaKernel's definition starts
   *
   * if @p line < @pre #signatureLineEnd_, then @post #bodyLineStart_ is set to #SRC_LINE_UNKNOWN
   * if @p line > @pre #bodyLineEnd_, then @post #bodyLineEnd_ is set to #SRC_LINE_UNKNOWN
   *
   * @post #signatureLineEnd_ <= #bodyLineStart <= #bodyLineEnd ||
   *       #bodyLineStart >= #signatureLineEnd_ && #bodyLineEnd_ == #SRC_LINE_UNKNOWN ||
   *       #bodyLineStart == #bodyLineEnd == #SRC_LINE_UNKNOWN
   */
  void setBodyLineStart(unsigned int line);

  /*
   * @brief Set the line number of which the body of this CudaKernel's definition ends
   */
  void setBodyLineEnd(unsigned int line);

  /*
   * @brief Set the start line and end line of this CudaKernel's body in source code
   *        Equivalent to calling #setBodyLineStart() followed by #setBodyLineEnd()
   */
  void setBodyLines(unsigned int start, unsigned int end);

  /*
   * @brief Retrieve the start line of this CudaKernel's body in source code
   * @return An integer >= 0 or SRC_LINE_UNKNOWN
   */
  int getBodyLineStart();

  /*
   * @brief Retrieve the end line of this CudaKernel's body in source code
   * @return An integer >= 0 or SRC_LINE_UNKNOWN
   */
  int getBodyLineEnd();

  /*
   * @brief Retrieve the number of lines this CudaKernel's body spans over (inclusive)
   * 
   * The number of lines is calculated as <code> 1 + #bodyLineEnd_ - #bodyLineStart_ </code>
   *
   * If any of #bodyLineStart_ or #signatureLbodyLineEnd have the value #SRC_LINE_UNKNOWN, then the
   * number of lines cannot be calculated and 0 is returned.
   * 
   * @return A positive integer or 0
   */
  int getBodyNumLines();

  /*
   * @brief Retrieve the first (source code) line of the kernel's definition
   * 
   * Equivalent to #getSignatureLineStart() 
   */
  int getLineStart();
  
  /*
   * @brief Retrieve the last (source code) line of the kernel's definition
   *
   * Equivalent to #getBodyLineEnd()
   */
  int getLineEnd();

  /*
   * @brief Retrieve the number of source code this kernel function spans
   *
   * If any of #signatureLineStart_ or bodyLineEnd_ == #SRC_LINE_UNKNOWN, then the number
   * of lines cannot be calculated and 0 is returned.
   *
   * @return A positive integer or 0
   */
  int getNumLines();
};

class CudaKernelLaunchConfiguration
{
  CudaKernelLaunchConfiguration();
  CudaKernelLaunchConfiguration(llvm::Value &grid,
                                llvm::Value &block);
  CudaKernelLaunchConfiguration(llvm::Value &grid,
                                llvm::Value &block,
                                llvm::Value &shMem);
  ~CudaKernelLaunchConfiguration() = default;

  llvm::Value *getGrid();
  llvm::Value *getBlock();
  llvm::Value *getSharedMemory();

  llvm::Value *getX(llvm::Value *dim3Value);
  llvm::Value *getY(llvm::Value *dim3Value);
  llvm::Value *getZ(llvm::Value *dim3Value);

};

class CudaKernelLaunch
{ 
  CudaKernelLaunch(CudaKernel &kernel, int line = -1);
  CudaKernelLaunch(CudaKernel &kernel, CudaKernelLaunchConfiguration &config, int line = -1);
  ~CudaKernelLaunch() = default;

  CudaKernelLaunchConfiguration *getLaunchConfigutation();

  void setLaunchConfiguration(CudaKernelLaunchConfiguration *config);

  void setLine(int line);

  int getLine();

  bool inLoop();

  bool inBranch();
};

} /* NAMESPACE kerma */

#endif /* KERMA_SUPPORT_CUDA_H */