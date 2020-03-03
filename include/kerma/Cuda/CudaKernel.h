/*
 * @file: include/kerma/Support/Cuda.h
 *
 * Includes various wrappers for CUDA related things
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
  unsigned int lineStart_;
  unsigned int lineEnd_;

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
   * Set the line number at which the kernel's definition starts
   */
  void setLineStart(unsigned int line);

  /*
   * Set the line number at which the kernel's definition ends (incluside)
   */
  void setLineEnd(unsigned int line);

  /*
   * Retrieve the first (source code) line of the kernel's definition
   */
  int getLineStart();
  
  /*
   * Retrieve the last (source code) line of the kernel's definition
   */
  int getLineEnd();

  /*
   * Retrieve the number of source code this kernel function spans
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