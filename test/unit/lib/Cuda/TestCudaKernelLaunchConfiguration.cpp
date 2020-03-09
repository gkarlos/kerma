#include "kerma/Cuda/CudaDim.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include <gtest/gtest.h>

#include <kerma/Cuda/CudaKernel.h>

using namespace kerma;

TEST( Constructor, Default )
{
  CudaKernelLaunchConfiguration config;
  ASSERT_EQ( config.getBlockIR(), nullptr);
  ASSERT_EQ( config.getGridIR(), nullptr);
  ASSERT_EQ( config.getSharedMemoryIR(), nullptr);
  ASSERT_EQ( config.getStreamIR(), nullptr);

  CudaDim emptyDim;

  ASSERT_EQ( config.getGrid(), emptyDim);
  ASSERT_EQ( config.getGrid(0), emptyDim.x);
  ASSERT_EQ( config.getGrid(1), emptyDim.y);
  ASSERT_EQ( config.getGrid(2), emptyDim.z);

  ASSERT_EQ( config.getBlock(), emptyDim);
  ASSERT_EQ( config.getBlock(0), emptyDim.x);
  ASSERT_EQ( config.getBlock(1), emptyDim.y);
  ASSERT_EQ( config.getBlock(2), emptyDim.z);

  ASSERT_EQ( config.getSharedMemory(), 0);
  ASSERT_EQ( config.getStream(), 0);
}

TEST( Constructor, GridBlockShmemStream )
{
  llvm::LoadInst dummy((llvm::Type *) nullptr, (llvm::Value *) nullptr);

  CudaKernelLaunchConfiguration config(&dummy, &dummy, &dummy, &dummy);

  ASSERT_EQ(&dummy, config.getBlockIR());
  ASSERT_EQ(&dummy, config.getGridIR());
  ASSERT_EQ(&dummy, config.getStreamIR());
  ASSERT_EQ(&dummy, config.getSharedMemoryIR());
}

TEST( IRValues, Set)
{
  CudaKernelLaunchConfiguration config;
  llvm::LoadInst dummy((llvm::Type *) nullptr, (llvm::Value *) nullptr);
  llvm::LoadInst dummy2((llvm::Type *) nullptr, (llvm::Value *) nullptr);

  config.setGridIR(&dummy);
  ASSERT_EQ(&dummy, config.getGridIR());
  config.setBlockIR(&dummy);
  ASSERT_EQ(&dummy, config.getBlockIR());
  config.setStreamIR(&dummy);
  ASSERT_EQ(&dummy, config.getStreamIR());
  config.setSharedMemoryIR(&dummy);
  ASSERT_EQ(&dummy, config.getSharedMemoryIR());

  config.setGridIR(&dummy2);
  ASSERT_EQ(&dummy2, config.getGridIR());
  config.setBlockIR(&dummy2);
  ASSERT_EQ(&dummy2, config.getBlockIR());
  config.setStreamIR(&dummy2);
  ASSERT_EQ(&dummy2, config.getStreamIR());
  config.setSharedMemoryIR(&dummy2);
  ASSERT_EQ(&dummy2, config.getSharedMemoryIR());
}

TEST( RealValues, SetGet)
{
  CudaKernelLaunchConfiguration config;

  CudaDim emptyDim;
  CudaDim nonemptyDim(1,2,3);

  config.setGrid(emptyDim);
  ASSERT_EQ(emptyDim, config.getGrid());
  config.setGrid(nonemptyDim);
  ASSERT_EQ(nonemptyDim, config.getGrid());
  config.setGrid(0, 10);
  ASSERT_EQ(config.getGrid(0), 10);
  config.setGrid(0, 20);
  ASSERT_EQ(config.getGrid(1), 20);
  config.setGrid(0, 30);
  ASSERT_EQ(config.getGrid(2), 30);


  config.setBlock(emptyDim);
  ASSERT_EQ(emptyDim, config.getBlock());
  config.setBlock(nonemptyDim);
  ASSERT_EQ(nonemptyDim, config.getBlock());
  config.setBlock(0, 10);
  ASSERT_EQ(config.getBlock(0), 10);
  config.setBlock(0, 20);
  ASSERT_EQ(config.getBlock(1), 20);
  config.setBlock(0, 30);
  ASSERT_EQ(config.getBlock(2), 30);

  config.setSharedMemory(1024);
  ASSERT_EQ(1024, config.getSharedMemory());
  config.setSharedMemory(2048);
  ASSERT_EQ(2048, config.getSharedMemory());

  config.setStream(15);
  ASSERT_EQ(15, config.getStream());
  config.setStream(25);
  ASSERT_EQ(25, config.getStream());
}