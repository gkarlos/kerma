#include "kerma/Base/Memory.h"
#include "kerma/NVVM/NVVM.h"

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>

using namespace kerma;

namespace {

TEST(MemoryTest, Init) {
  Memory Memory(0, "mem1", nvvm::AddressSpace::Generic, Dim(2,2,2), true);
  ASSERT_EQ(Memory.getID(), 0);
  ASSERT_EQ(Memory.getName().compare("mem1"), 0);
  ASSERT_EQ(Memory.getAddrSpace(), nvvm::AddressSpace::Generic);
  ASSERT_EQ(Memory.getDim(), Dim(2,2,2));
  ASSERT_EQ(Memory.getPos(), Memory::Unknown);
  ASSERT_TRUE(Memory.dimIsAssumed());
}

TEST(MemoryTest, IDs) {
  std::vector<unsigned int> IDs;
  for ( int i = 0; i < 50; ++i)
    IDs.push_back(Memory("mem1", nvvm::AddressSpace::Generic, Dim(2,2,2), true).getID());
  for ( int i = 0; i < 50; ++i)
    ASSERT_FALSE(std::count(IDs.begin(), IDs.end(), IDs[i]) > 1);
}

}