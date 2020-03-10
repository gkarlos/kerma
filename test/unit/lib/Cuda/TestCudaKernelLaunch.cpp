#include "kerma/Cuda/Cuda.h"
#include "kerma/Support/SourceCode.h"
#include "llvm/ADT/Twine.h"
#include <gtest/gtest.h>


#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Cuda/CudaModule.h>

#include "llvm/Support/SourceMgr.h"
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>

using namespace llvm;
using namespace kerma;

#define POLYBENCH std::string("../../../../../examples/polybench/cuda/")

TEST( Constructor, CudaKernel_int)
{

  std::string deviceIR = POLYBENCH + "2MM/2mm-cuda-nvptx64-nvidia-cuda-sm_52.ll";
  LLVMContext context;
  SMDiagnostic error;
  auto deviceModule = parseIRFile(deviceIR, error, context);
  CudaModule module(*deviceModule.get(), *deviceModule.get());


  llvm::Function *f = Function::Create( (FunctionType *) nullptr,
                                        GlobalValue::InternalLinkage,
                                        "hello",
                                        deviceModule.get()
                                      );
  
  CudaKernel kernel(*f, CudaSide::DEVICE);
  CudaKernelLaunch launch(kernel);
  CudaKernelLaunchConfiguration emptyConfig;

  ASSERT_EQ(launch.getLine(), SRC_LINE_UNKNOWN);
  ASSERT_EQ(launch.getLaunchConfigutation(), emptyConfig);
  ASSERT_EQ(launch.getKernel(), kernel);
  ASSERT_EQ(launch.inLoop(), false);
  ASSERT_EQ(launch.inBranch(), false);
  ASSERT_EQ(launch.inThenBranch(), false);
  ASSERT_EQ(launch.inElseBranch(), false);


  CudaKernelLaunch launch2(kernel, 152);
  ASSERT_EQ(launch.getLine(), 152);
}