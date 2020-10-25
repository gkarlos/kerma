#include "kerma/Base/MemoryAccess.h"
#include "kerma/Base/Memory.h"

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>

using namespace kerma;

namespace {

TEST(MemoryAccessTest, Init) {
  Memory Memory(0, "mem1", nvvm::AddressSpace::Generic, Dim(2,2,2), true);
  MemoryAccess Access(0, Memory, nullptr, MemoryAccess::Load);
  ASSERT_EQ(Access.getType(), MemoryAccess::Load);
  ASSERT_EQ(Access.getID(), 0);
  ASSERT_EQ(Access.getMemory(), Memory);
  ASSERT_EQ(Access.getPtr(), nullptr);
}

TEST(MemoryAccessTest, IDs) {
  std::vector<unsigned int> IDs;
  Memory Memory(0, "mem1", nvvm::AddressSpace::Generic, Dim(2,2,2), true);

  for ( int i = 0; i < 50; ++i)
    IDs.push_back(MemoryAccess(0, Memory, nullptr, MemoryAccess::Load).getID());
  for ( int i = 0; i < 50; ++i)
    ASSERT_FALSE(std::count(IDs.begin(), IDs.end(), IDs[i]) > 1);
}


}