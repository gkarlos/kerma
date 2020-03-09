#include "kerma/Cuda/CudaDim.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include "llvm/IR/Value.h"
#include <kerma/Cuda/CudaKernel.h>
#include <kerma/Support/Util.h>

namespace kerma 
{

CudaKernelLaunchConfiguration::CudaKernelLaunchConfiguration()
: CudaKernelLaunchConfiguration(nullptr, nullptr, nullptr, nullptr)
{}

CudaKernelLaunchConfiguration::CudaKernelLaunchConfiguration(llvm::Value *grid,
                                                             llvm::Value *block)
: CudaKernelLaunchConfiguration(grid, block, nullptr, nullptr)
{}

CudaKernelLaunchConfiguration::CudaKernelLaunchConfiguration(llvm::Value *grid,
                                                             llvm::Value *block,
                                                             llvm::Value *shmem)
: CudaKernelLaunchConfiguration(grid, block, shmem, nullptr)
{}

CudaKernelLaunchConfiguration::CudaKernelLaunchConfiguration(llvm::Value *grid,
                                                             llvm::Value *block,
                                                             llvm::Value *shmem,
                                                             llvm::Value *stream)
: gridIR_(grid),
  blockIR_(block),
  shmemIR_(shmem),
  streamIR_(stream),
  shmemReal_(0),
  streamReal_(0)
{}

static int _X_ = 0;
static int _Y_ = 1;
static int _Z_ = 2;
static int _UNKNOWN_ = -1;

void
CudaKernelLaunchConfiguration::operator=(const CudaKernelLaunchConfiguration &other)
{
  gridIR_ = other.gridIR_;
  blockIR_ = other.blockIR_;
  shmemIR_ = other.shmemIR_;
  streamIR_ = other.streamIR_;
  gridReal_ = other.gridReal_;
  blockReal_ = other.blockReal_;
  shmemReal_ = other.shmemReal_;
  streamReal_ = other.streamReal_;
}

#define BOTH_NULL_OR_NON_NULL(a,b) ((a == nullptr && b == nullptr) || (a != nullptr && b != nullptr))

bool
CudaKernelLaunchConfiguration::operator==(const CudaKernelLaunchConfiguration &other)
{
  
  return gridIR_ == other.gridIR_ 
      && blockIR_ == other.blockIR_
      && shmemIR_ == other.shmemIR_
      && streamIR_ == other.streamIR_
      && gridReal_ == other.gridReal_
      && blockReal_ == other.blockReal_
      && shmemReal_ == other.shmemReal_
      && streamReal_ == other.streamReal_;
}

// Grid Stuff

void
CudaKernelLaunchConfiguration::setGridIR(llvm::Value *grid) { gridIR_ = grid; }

llvm::Value *
CudaKernelLaunchConfiguration::getGridIR() { return gridIR_; }

/// TODO @todo Return an llvm::Value for the dim-th dimension of the Grid
llvm::Value *
CudaKernelLaunchConfiguration::getGridIR(unsigned int dim)
{
  NOT_IMPLEMENTED_YET;

  if ( dim > _Z_ )
    return nullptr;
  else {
    return nullptr;
  }
}

void
CudaKernelLaunchConfiguration::setGrid(CudaDim &grid) { gridReal_ = grid; }

void
CudaKernelLaunchConfiguration::setGrid(unsigned int dim, unsigned int value)
{
  if ( dim == _X_ )
    gridReal_.x = value;
  else if ( dim == _Y_ )
    gridReal_.y = value;
  else if ( dim == _Z_ )
    gridReal_.z = value;
}

void
CudaKernelLaunchConfiguration::setGrid(unsigned int x, unsigned int y, unsigned int z)
{
  gridReal_.x = x;
  gridReal_.y = y;
  gridReal_.z = z;
}

CudaDim &
CudaKernelLaunchConfiguration::getGrid() { return gridReal_; }

int
CudaKernelLaunchConfiguration::getGrid(unsigned int dim)
{
  if ( dim == _X_ )
    return gridReal_.x;
  else if ( dim == _Y_ )
    return gridReal_.y;
  else if ( dim == _Z_ )
    return gridReal_.z;
  else
    return _UNKNOWN_;
}

// Block Stuff

void
CudaKernelLaunchConfiguration::setBlockIR(llvm::Value *block) { blockIR_ = block; }

llvm::Value *
CudaKernelLaunchConfiguration::getBlockIR() { return blockIR_; }

/// TODO @todo Return an llvm::Value for the dim-th dimension of the Block
llvm::Value *
CudaKernelLaunchConfiguration::getBlockIR(unsigned int dim)
{
  NOT_IMPLEMENTED_YET;

  if ( dim > _Z_ )
    return nullptr;
  else {
    return nullptr;
  }
}

void
CudaKernelLaunchConfiguration::setBlock(CudaDim &block) { blockReal_ = block; }

void
CudaKernelLaunchConfiguration::setBlock(unsigned int dim, unsigned int value)
{
  if ( dim == _X_ )
    blockReal_.x = value;
  else if ( dim == _Y_ )
    blockReal_.y = value;
  else if ( dim == _Z_ )
    blockReal_.z = value;
}

void
CudaKernelLaunchConfiguration::setBlock(unsigned int x, unsigned int y, unsigned int z)
{
  blockReal_.x = x;
  blockReal_.y = y;
  blockReal_.z = z;
}

CudaDim &
CudaKernelLaunchConfiguration::getBlock() { return blockReal_; }

int
CudaKernelLaunchConfiguration::getBlock(unsigned int dim)
{
   if ( dim == _X_ )
    return blockReal_.x;
  else if ( dim == _Y_ )
    return blockReal_.y;
  else if ( dim == _Z_ )
    return blockReal_.z;
  else
    return _UNKNOWN_;
}

void
CudaKernelLaunchConfiguration::setSharedMemoryIR(llvm::Value *sharedMemory) { shmemIR_ = sharedMemory; }

llvm::Value *
CudaKernelLaunchConfiguration::getSharedMemoryIR() { return shmemIR_; }

void
CudaKernelLaunchConfiguration::setSharedMemory(unsigned int value) { shmemReal_ = value; }

int
CudaKernelLaunchConfiguration::getSharedMemory() { return shmemReal_; }

void
CudaKernelLaunchConfiguration::setStreamIR(llvm::Value *stream) { streamIR_ = stream; }

llvm::Value *
CudaKernelLaunchConfiguration::getStreamIR() { return streamIR_; }

void
CudaKernelLaunchConfiguration::setStream(unsigned int value) { streamReal_ = value; }

int
CudaKernelLaunchConfiguration::getStream() { return streamReal_; }


} /* NAMESPACE kerma */