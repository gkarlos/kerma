/*
 * @file: include/Cuda/CudaKernel.h
 */
#ifndef KERMA_SUPPORT_CUDA_H
#define KERMA_SUPPORT_CUDA_H

#include "kerma/Support/PrettyPrintable.h"
#include "kerma/Support/SourceCode.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <kerma/Cuda/Cuda.h>

#if (__cplusplus >= 201703L) /// >= C++17
  #include <optional>
  #define OPTIONAL std::optional
#else
  #include <experimental/optional>
  #define OPTIONAL std::experimental::optional
#endif

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

/*
 * This class represents a Cuda kernel launch configuration
 * It includes information about grid size, block size, amount of shared memory
 * and the cuda stream the kernel is launched on.
 *
 * This class is meant to be used by CudaKudaKernelLaunch
 */
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


/*
 * This class represents a Cuda kernel launch
 * It is meant to be populated by DetectKernelLaunchesPass
 */
class CudaKernelLaunch
{ 
public:
  CudaKernelLaunch(CudaKernel &kernel, int line = SRC_LINE_UNKNOWN);
  CudaKernelLaunch(CudaKernel &kernel, CudaKernelLaunchConfiguration *config, int line = SRC_LINE_UNKNOWN);
  ~CudaKernelLaunch();

/// API
public:
  /*
   * @brief Retrieve the CudaKernel this launch is relevant to
   */
  CudaKernel & getKernel();

  /*
   * @brief Retrieve the launch configuration associated with this launch
   */
  CudaKernelLaunchConfiguration *getLaunchConfigutation();

  /*
   * @brief Set a launch configuration for this launch
   */
  void setLaunchConfiguration(CudaKernelLaunchConfiguration *config);

  /*
   * @brief Set the CallInst for the cudaLaunchKernel() call associated with this launch
   *
   * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c
   */
  void setCudaLaunchKernelCall(llvm::CallInst *kernelCall);

  /*
   * @brief Retrieve the CallInst for the cudaLaunchKernel() call associated with this launch
   */
  llvm::CallInst* getCudaLaunchKernelCall();

  /*
   * @brief Get the source code line where the kernel is launched
   */
  int getLine();

  /*
   * @brief Get whether this launch is within a loop
   */
  OPTIONAL<bool> inLoop();

  /*
   * @brief Set whether this launch is within a loop
   */
  void setInLoop(bool inLoop);

  /*
   * @brief Clear the inLoop status of this launch.
   * @post  The inLoop value is unknown (neither true nor false)
   */
  void unsetInLoop();

  /*
   * @brief Get whether this launch is within the 'then' branch of an if statement
   */
  OPTIONAL<bool> inThenBranch();

  /*
   * @brief Check whether this launch is within the 'else' branch of an if statemnt
   */
  OPTIONAL<bool> inElseBranch();

  /*
   * @brief Check whether this launch is within an if statement (i.e either in the 'then'
   *    or the 'else' part of the if statement)
   */
  OPTIONAL<bool> inBranch();

  /*
   * @brief Set whether this launch is within the 'then' branch of an if statement
   * @post  if inThenBranch == true then inElseBranch == false
   */
  void setInThenBranch(bool inThenBranch);

  /*
   * @brief Set whether this launch is within the 'else' branch of an if statement
   * @post  if inElseBranch == true then inThenBranch == false
   */
  void setInElseBranch(bool inElseBranch);

  /*
   * @brief Clear the inBranch status of this launch.
   * @post  inThenBranch, inElseBranch is unknown (neither true nor false)
   */
  void unsetInBranch();

private:
  CudaKernel &kernel_;
  CudaKernelLaunchConfiguration launchConfiguration_;
  llvm::CallInst *cudaLaunchKernelCall_;
  OPTIONAL<bool> inLoop_;
  OPTIONAL<bool> inThen_;
  OPTIONAL<bool> inElse_;
  int line_;
};

} /* NAMESPACE kerma */

#endif /* KERMA_SUPPORT_CUDA_H */